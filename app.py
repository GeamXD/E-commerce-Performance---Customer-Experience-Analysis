import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .prediction-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Define a cached function to load model artifacts outside the class
@st.cache_resource
def load_model_artifacts():
    """Load the trained model artifacts"""
    try:
        artifacts = joblib.load('churn_model_artifacts.pkl')
        return artifacts
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first and ensure 'churn_model_artifacts.pkl' is in the same directory.")
        return None

class StreamlitChurnPredictor:
    def __init__(self):
        # Initialize model_artifacts by calling the cached function
        self.model_artifacts = load_model_artifacts()
    
    def predict_churn(self, customer_data):
        """Make churn prediction"""
        if self.model_artifacts is None:
            return None, None
        
        try:
            # Get model components
            model = self.model_artifacts['model']
            scaler = self.model_artifacts['scaler']
            label_encoders = self.model_artifacts['label_encoders']
            feature_columns = self.model_artifacts['feature_columns']
            numeric_features = self.model_artifacts['numeric_features']
            
            # Prepare input data
            input_data = customer_data.copy()
            
            # Encode categorical variables
            for col, encoder in label_encoders.items():
                if col in input_data.columns:
                    # Ensure the column is treated as string for encoding
                    input_data[col] = input_data[col].astype(str)
                    
                    # Handle unseen categories by checking if the value is in known classes
                    # If not, map to the first class or a designated 'unknown' category
                    input_data[col] = input_data[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else encoder.transform([encoder.classes_[0]])[0]
                    )
            
            # Ensure all features are present and in the correct order
            # Fill missing feature columns with 0 (or a sensible default for your model)
            for col in feature_columns:
                if col not in input_data.columns:
                    # Use a sensible default for missing features; 0 might not always be appropriate
                    # for all feature types (e.g., categorical). Adjust as needed based on your model training.
                    input_data[col] = 0
            
            # Reorder columns to match the training order
            input_data = input_data[feature_columns]
            
            # Scale numeric features
            input_scaled = input_data.copy()
            # Ensure only numeric features are scaled
            input_scaled[numeric_features] = scaler.transform(input_data[numeric_features])
            
            # Make prediction
            churn_probability = model.predict_proba(input_scaled)[:, 1]
            churn_prediction = model.predict(input_scaled)
            
            return churn_prediction[0], churn_probability[0]
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.warning("Please ensure that 'churn_model_artifacts.pkl' contains all necessary model components (model, scaler, label_encoders, feature_columns, numeric_features) and that input features match the training schema.")
            return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize predictor
    predictor = StreamlitChurnPredictor()
    
    # Sidebar for input
    st.sidebar.header("📝 Customer Information")
    
    # Customer behavior inputs
    st.sidebar.subheader("Purchase Behavior")
    recency_days = st.sidebar.slider("Days Since Last Purchase", 0, 500, 30)
    frequency = st.sidebar.slider("Number of Orders", 1, 50, 5)
    monetary = st.sidebar.slider("Total Spent ($)", 0.0, 5000.0, 200.0)
    
    # Review and satisfaction
    st.sidebar.subheader("Customer Satisfaction")
    avg_review_score = st.sidebar.slider("Average Review Score", 1.0, 5.0, 4.0, 0.1)
    review_score_std = st.sidebar.slider("Review Score Variability", 0.0, 2.0, 0.5, 0.1)
    
    # Payment behavior
    st.sidebar.subheader("Payment Information")
    payment_type = st.sidebar.selectbox("Preferred Payment Type", 
                                       ["credit_card", "boleto", "voucher", "debit_card"])
    avg_installments = st.sidebar.slider("Average Installments", 1.0, 24.0, 2.0)
    avg_payment_value = st.sidebar.slider("Average Payment Value ($)", 0.0, 1000.0, 100.0)
    
    # Geographic information
    st.sidebar.subheader("Location")
    customer_state = st.sidebar.selectbox("Customer State", 
                                         ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "GO", "ES"])
    customer_city = st.sidebar.text_input("Customer City", "São Paulo")
    
    # Product preferences
    st.sidebar.subheader("Product Preferences")
    product_category = st.sidebar.selectbox("Most Purchased Category", 
                                           ["bed_bath_table", "health_beauty", "sports_leisure", 
                                            "computers_accessories", "furniture_decor", "watches_gifts",
                                            "telephony", "auto", "toys", "cool_stuff"])
    
    # Advanced features
    with st.sidebar.expander("Advanced Features"):
        avg_price = st.slider("Average Product Price ($)", 0.0, 500.0, 50.0)
        avg_freight = st.slider("Average Freight Cost ($)", 0.0, 100.0, 15.0)
        avg_product_weight = st.slider("Average Product Weight (g)", 0.0, 5000.0, 500.0)
        avg_delivery_days = st.slider("Average Delivery Days", 0, 30, 10)
    
    # Prediction button (only one button needed)
    predict_button = st.sidebar.button("🔮 Predict Churn", type="primary")
    
    if predict_button:
        # Prepare input data
        customer_data = pd.DataFrame({
            'recency_days': [recency_days],
            'frequency': [frequency],
            'monetary': [monetary],
            'review_score_mean': [avg_review_score],
            'review_score_std': [review_score_std],
            'payment_type': [payment_type],
            'payment_installments_mean': [avg_installments],
            'payment_value_mean': [avg_payment_value],
            'customer_state': [customer_state],
            'customer_city': [customer_city],
            'product_category_name_english': [product_category],
            'price_mean': [avg_price],
            'freight_value_mean': [avg_freight],
            'product_weight_g_mean': [avg_product_weight],
            'order_status': ['delivered'] # Assuming 'delivered' is a valid category for the model
        })
        
        # Make prediction
        prediction, probability = predictor.predict_churn(customer_data)
        
        if prediction is not None:
            # Main content area
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Prediction result
                if prediction == 1:
                    st.markdown(f"""
                    <div class="metric-card prediction-high">
                        <h2 style="color: #f44336; text-align: center;">⚠️ HIGH CHURN RISK</h2>
                        <h3 style="text-align: center;">Churn Probability: {probability:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommendations for high-risk customers
                    st.subheader("🎯 Recommended Actions")
                    st.write("This customer is at high risk of churning. Consider:")
                    st.write("• Personalized retention offers")
                    st.write("• Proactive customer service outreach")
                    st.write("• Loyalty program enrollment")
                    st.write("• Product recommendations based on purchase history")
                    
                else:
                    st.markdown(f"""
                    <div class="metric-card prediction-low">
                        <h2 style="color: #4caf50; text-align: center;">✅ LOW CHURN RISK</h2>
                        <h3 style="text-align: center;">Churn Probability: {probability:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommendations for low-risk customers
                    st.subheader("🎯 Recommended Actions")
                    st.write("This customer is likely to stay. Consider:")
                    st.write("• Cross-selling complementary products")
                    st.write("• Referral program incentives")
                    st.write("• Premium service offerings")
                    st.write("• Collecting feedback for service improvement")
            
            # Probability gauge
            st.subheader("📊 Churn Probability Gauge")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature analysis
            st.subheader("📈 Customer Profile Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Recency Score", 
                         "High Risk" if recency_days > 90 else "Good", 
                         f"{recency_days} days")
                
            with col2:
                st.metric("Frequency Score", 
                         "Low" if frequency <= 2 else "Good", 
                         f"{frequency} orders")
                
            with col3:
                st.metric("Monetary Score", 
                         "Low" if monetary < 100 else "Good", 
                         f"${monetary:.2f}")
            
            # Risk factors
            st.subheader("⚠️ Risk Factors Identified")
            risk_factors = []
            
            if recency_days > 180:
                risk_factors.append("🔴 Very long time since last purchase")
            elif recency_days > 90:
                risk_factors.append("🟡 Long time since last purchase")
                
            if frequency <= 2:
                risk_factors.append("🔴 Low purchase frequency")
            elif frequency <= 5:
                risk_factors.append("🟡 Moderate purchase frequency")
                
            if monetary < 50:
                risk_factors.append("🔴 Low total spending")
            elif monetary < 100:
                risk_factors.append("🟡 Moderate total spending")
                
            if avg_review_score < 3:
                risk_factors.append("🔴 Low satisfaction scores")
            elif avg_review_score < 4:
                risk_factors.append("🟡 Moderate satisfaction scores")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.write("🟢 No major risk factors identified")
    
    # Instructions displayed only if prediction hasn't been made
    else: # This else block executes when the predict_button is not clicked
        st.info("👈 Please fill in the customer information in the sidebar and click 'Predict Churn' to get started.")
        
        # Sample data explanation
        st.subheader("📋 How to Use This Tool")
        st.write("""
        1. **Customer Behavior**: Enter recent purchase patterns (recency, frequency, monetary value)
        2. **Satisfaction**: Provide review scores and feedback data
        3. **Payment Info**: Select payment preferences and patterns
        4. **Demographics**: Choose location and product preferences
        5. **Advanced Features**: Fine-tune with additional product and delivery metrics
        6. **Get Prediction**: Click the predict button to see churn probability and recommendations
        """)
        
        # Model information
        with st.expander("ℹ️ About the Model"):
            st.write("""
            This churn prediction model uses **XGBoost** (Extreme Gradient Boosting) trained on customer behavior data.
            
            **Key Features:**
            - RFM Analysis (Recency, Frequency, Monetary)
            - Customer Satisfaction Scores
            - Payment and Product Preferences
            - Geographic Information
            - Advanced Feature Engineering
            
            **Model Performance:**
            - Hyperparameter tuned using RandomizedSearchCV
            - Cross-validated for optimal performance
            - Balanced approach to precision and recall
            
            **Churn Definition:**
            A customer is considered churned if they meet any of these criteria:
            - No purchase in 6+ months with ≤2 total orders
            - Low satisfaction (avg review < 3) + no purchase in 3+ months
            - Low spending (<$50) + no purchase in 4+ months
            """)

if __name__ == "__main__":
    main()