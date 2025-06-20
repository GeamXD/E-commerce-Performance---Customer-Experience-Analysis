import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model persistence
import joblib


class CustomerChurnPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.label_encoders = {}

    def create_churn_target(self, df):
        """
        Create churn target variable based on customer behavior patterns
        """
        print("Creating churn target variable...")

        # Convert date columns to datetime
        date_columns = ['review_creation_date', 'order_purchase_timestamp',
                        'order_approved_at', 'order_delivered_customer_date']

        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Calculate recency (days since last order)
        max_date = df['order_purchase_timestamp'].max()
        customer_last_order = df.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
        customer_last_order['recency_days'] = (max_date - customer_last_order['order_purchase_timestamp']).dt.days

        # Calculate frequency (number of orders per customer)
        customer_frequency = df.groupby('customer_unique_id').size().reset_index(name='frequency')

        # Calculate monetary (total spent per customer)
        customer_monetary = df.groupby('customer_unique_id')['payment_value'].sum().reset_index()
        customer_monetary.rename(columns={'payment_value': 'monetary'}, inplace=True)

        # Calculate average review score per customer
        customer_review = df.groupby('customer_unique_id')['review_score'].mean().reset_index()
        customer_review.rename(columns={'review_score': 'avg_review_score'}, inplace=True)

        # Merge all customer metrics
        customer_metrics = customer_last_order.merge(customer_frequency, on='customer_unique_id')
        customer_metrics = customer_metrics.merge(customer_monetary, on='customer_unique_id')
        customer_metrics = customer_metrics.merge(customer_review, on='customer_unique_id')

        # Define churn based on business rules:
        # Churn = 1 if:
        # - Recency > 180 days (6 months) AND frequency <= 2 OR
        # - Average review score < 3 AND recency > 90 days OR
        # - Monetary value < 50 AND recency > 120 days

        customer_metrics['churn'] = 0

        churn_condition = (
                ((customer_metrics['recency_days'] > 180) & (customer_metrics['frequency'] <= 2)) |
                ((customer_metrics['avg_review_score'] < 3) & (customer_metrics['recency_days'] > 90)) |
                ((customer_metrics['monetary'] < 50) & (customer_metrics['recency_days'] > 120))
        )

        customer_metrics.loc[churn_condition, 'churn'] = 1

        print(f"Churn distribution:")
        print(customer_metrics['churn'].value_counts())
        print(f"Churn rate: {customer_metrics['churn'].mean():.2%}")

        return customer_metrics

    def prepare_features(self, df, customer_metrics):
        """
        Prepare features for modeling
        """
        print("Preparing features...")

        # Aggregate features at customer level
        customer_features = df.groupby('customer_unique_id').agg({
            'review_score': ['mean', 'std', 'min', 'max'],
            'payment_installments': ['mean', 'max'],
            'payment_value': ['sum', 'mean', 'std'],
            'price': ['sum', 'mean', 'std'],
            'freight_value': ['sum', 'mean'],
            'product_name_lenght': 'mean',
            'product_description_lenght': 'mean',
            'product_photos_qty': 'mean',
            'product_weight_g': 'mean',
            'product_length_cm': 'mean',
            'product_height_cm': 'mean',
            'product_width_cm': 'mean',
            'customer_latitude': 'first',
            'customer_longitude': 'first',
            'seller_latitude': 'mean',
            'seller_longitude': 'mean'
        }).reset_index()

        # Flatten column names
        customer_features.columns = ['customer_unique_id'] + [f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
                                                              for col in customer_features.columns[1:]]

        # Add categorical features
        categorical_features = df.groupby('customer_unique_id').agg({
            'payment_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
            'customer_state': 'first',
            'customer_city': 'first',
            'product_category_name_english': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
            'order_status': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        }).reset_index()

        # Merge all features
        features = customer_features.merge(categorical_features, on='customer_unique_id')
        features = features.merge(
            customer_metrics[['customer_unique_id', 'recency_days', 'frequency', 'monetary', 'churn']],
            on='customer_unique_id')

        # Handle missing values
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())

        categorical_cols = features.select_dtypes(include=[object]).columns
        categorical_cols = [col for col in categorical_cols if col != 'customer_unique_id']
        features[categorical_cols] = features[categorical_cols].fillna('unknown')

        return features

    def build_model(self, features):
        """
        Build and train XGBoost model with hyperparameter tuning
        """
        print("Building XGBoost model...")

        # Prepare features and target
        X = features.drop(['customer_unique_id', 'churn'], axis=1)
        y = features['churn']

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=[object]).columns.tolist()

        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', 'passthrough', categorical_features)
            ])

        # Encode categorical variables
        X_encoded = X.copy()
        for col in categorical_features:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale numeric features only
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

        # Define XGBoost model
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss'
        )

        # Hyperparameter tuning
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1]
        }

        print("Performing hyperparameter tuning...")
        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_distributions,
            n_iter=50,
            cv=5,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        random_search.fit(X_train_scaled, y_train)

        # Best model
        best_model = random_search.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        # Evaluation
        print(f"\nBest parameters: {random_search.best_params_}")
        print(f"Best cross-validation score: {random_search.best_score_:.4f}")
        print(f"\nTest set performance:")
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Feature Importances:")
        print(feature_importance.head(10))

        # Store model components
        self.model = best_model
        self.scaler = scaler
        self.feature_columns = X_encoded.columns.tolist()
        self.numeric_features = numeric_features

        # Save model
        model_artifacts = {
            'model': best_model,
            'scaler': scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'numeric_features': numeric_features
        }

        joblib.dump(model_artifacts, 'churn_model_artifacts.pkl')
        print("Model saved as 'churn_model_artifacts.pkl'")

        return best_model, feature_importance

    def predict_churn(self, customer_data):
        """
        Predict churn for new customer data
        """
        if self.model is None:
            # Load model if not already loaded
            artifacts = joblib.load('churn_model_artifacts.pkl')
            self.model = artifacts['model']
            self.scaler = artifacts['scaler']
            self.label_encoders = artifacts['label_encoders']
            self.feature_columns = artifacts['feature_columns']
            self.numeric_features = artifacts['numeric_features']

        # Prepare input data
        input_data = customer_data.copy()

        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in input_data.columns:
                # Handle unseen categories
                try:
                    input_data[col] = encoder.transform(input_data[col].astype(str))
                except ValueError:
                    # If unseen category, use the most frequent class
                    input_data[col] = encoder.transform([encoder.classes_[0]])[0]

        # Ensure all features are present
        for col in self.feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns
        input_data = input_data[self.feature_columns]

        # Scale numeric features
        input_scaled = input_data.copy()
        input_scaled[self.numeric_features] = self.scaler.transform(input_data[self.numeric_features])

        # Predict
        churn_probability = self.model.predict_proba(input_scaled)[:, 1]
        churn_prediction = self.model.predict(input_scaled)

        return churn_prediction, churn_probability


# Usage example (assuming you have loaded your dataset as 'df')
def main():
    # Load your dataset
    # df = pd.read_csv('your_dataset.csv')

    # Initialize predictor
    predictor = CustomerChurnPredictor()

    # Create churn target
    # customer_metrics = predictor.create_churn_target(df)

    # Prepare features
    # features = predictor.prepare_features(df, customer_metrics)

    # Build and train model
    # model, feature_importance = predictor.build_model(features)

    print("Customer Churn Prediction Model Ready!")
    print("Next step: Run the Streamlit app for predictions")


if __name__ == "__main__":
    main()