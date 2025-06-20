# E-commerce Performance & Customer Experience Analysis
- Author: Musa Godwin
- Date: June 20, 2024
- Tools Used: Python (Data Cleaning, Preprocessing, ML Modeling), Tableau (Visualization & Dashboarding)

## Executive Summary
This report provides a comprehensive analysis of the company's Q4 performance, revealing a critical disconnect between sales acquisition and post-purchase customer experience. While lead conversion and top-performer sales show strength, these gains are being nullified by systemic issues in logistics and fulfillment.

- The Core Problem: A significant 21.4% decline in Customer Lifetime Value (CLV) and a 21.38% drop in total sales are directly linked to a deteriorating delivery experience. Despite an improvement in average delivery speed, a decrease in on-time reliability and a spike in delays in key regions have led to a 4.72% drop in average review scores and a catastrophic 95.6% churn rate.


- The Path Forward: Our strategy must pivot from aggressive acquisition to aggressive retention. This report outlines pointed recommendations focused on three pillars:


- Fixing the Fulfillment Engine: Overhauling logistics to prioritize reliability over raw average speed.


- Activating Customer Retention: Implementing targeted programs to re-engage at-risk customers and reward loyalty.


- Capitalizing on Bright Spots: Replicating the success of top sellers and products to de-risk the business.

## Key Performance Indicators (KPIs) - The Story in Numbers

### Customer & Retention Metrics
- Customer Lifetime Value (CLV): $191.74 (🔻 -21.4% vs. last quarter). A major decline, indicating we are extracting significantly less value from each customer.


- Customer Retention Rate: 4.4% (🔼 +6.9% vs. last quarter). While showing a slight improvement, a rate this low is unsustainable and indicates a fundamental "leaky bucket" problem.


- Customer Churn Rate: 95.6% (🔽 -0.3% vs. last quarter). The marginal improvement is statistically insignificant. A 95.6% churn rate is an existential threat to the business model.

### Sales & Revenue Metrics
- Total Sales: $1.61M (🔻 -21.38% vs. last quarter). Aligns with the drop in CLV, confirming a broad-based revenue decline.


- Top 5 Products (by Avg. Revenue): Generated $187.07k in revenue (🔼 +145.33% vs. last quarter). This highlights a heavy reliance on a few high-ticket categories (Computers, Agro, Small Appliances).


- Top 5 Sellers: Generated $151.85k in revenue (🔼 +4891% vs. last quarter). This astronomical growth points to a few hyper-performing sellers carrying a disproportionate amount of the business.

### Funnel & Operations Metrics
- Lead Conversion Rate: 100%. This metric is likely flawed or mis-defined. A 100% conversion rate is improbable and suggests leads are only being tracked after the point of conversion. This needs investigation.


- Average Time to Close: 20 days. A baseline metric to monitor as we adjust marketing and sales strategies.


- Average Delivery Time: 12 days (Improved by 4 days). A positive "vanity metric" that masks the real problem.


- On-Time Delivery (OTD) Rate: 92.27% (🔻 -2.66% vs. last quarter). This is a critical failure. We are faster on average but less reliable, which breaks customer trust.


- Average Delivery Delay: 0.6 days (Worsened by 0.4 days). This increase, combined with the OTD drop, is the root of the satisfaction problem.

- Top 5 Cities Delivery Delay: Averaged 8 days (🔼 +202.5% vs. last quarter). A massive operational failure in specific geographies is severely damaging our brand reputation there.

### Customer Satisfaction Metrics
- Average Review Score: 4.1 (🔻 -4.72% vs. last quarter). A significant drop, moving from a "Good" to a "Mediocre" rating in the eyes of the customer.


- Correlation (Delivery Delay vs. Review Score): -0.29 (Strengthened negativity by -119%). This is the smoking gun. The statistical link between late deliveries and poor reviews has more than doubled in strength this quarter.

## Recommendations Based on Findings
### 1. Declare War on Delivery Delays (Highest Priority)
- The data is unequivocal: logistics failures are the primary driver of customer dissatisfaction and churn.
- Action: Immediately segment logistics partners by performance in the top 5 worst-performing cities (starting with Formosa). Renegotiate SLAs to include penalties for missing OTD targets, not just average speed. Explore partnerships with local, more reliable couriers in these problem areas.
- Action: Implement a proactive communication system. If a delivery is projected to be late, automatically notify the customer with a sincere apology and a small store credit for their next purchase. This turns a negative experience into a retention opportunity.
- Action: Publicly feature an "On-Time & Reliable" badge on product pages for items fulfilled through our top-performing logistics partners. Use this as a selling point.

### 2. Launch a Targeted Customer Retention & Win-Back Program
- With 95.6% churn, acquisition is inefficient. We must plug the leak.

- Action: Use the predictive model to identify customers who have received a delayed order but have not yet churned. Target them immediately with a "We're Sorry & We're Improving" campaign, offering a compelling discount to encourage a second purchase under our improved logistics.

- Action: Create a loyalty program for customers who purchase from our high-satisfaction product categories (Fashion, Flowers, Books), as they are our happiest cohort. Encourage them to become brand advocates.

- Action: For high-CLV customers who have churned, initiate a personal outreach from a customer success manager to understand their issues and offer a significant incentive to return.

### 3. Diversify and Replicate Success to De-risk the Business
- Over-reliance on a few sellers and products is a major risk.
- Action: Conduct a qualitative analysis of the top 5 sellers. What are they doing differently? (Product selection, customer communication, marketing). Codify their successful strategies into a training program for all other sellers.
- Action: Cross-promote high-satisfaction products (e.g., Flowers, Fashion) to customers who have purchased high-revenue products (e.g., Computers, Appliances). This can increase overall account satisfaction and introduce customers to a better product experience.


## A/B Testing Opportunities
- To validate our recommendations with data, we should implement the following tests:

#### A. Funnel/Experience Level
- Hypothesis: Proactively notifying customers about a delivery delay with a $5 store credit will reduce the negative impact on their subsequent review score and increase the probability of a second purchase compared to not notifying them.
- Control Group (A): Customers with a delayed order receive no special communication.
- Test Group (B): Customers with a delayed order receive an automated email and SMS with an apology and a $5 coupon code.
- Primary Metric: Average review score from these customers.
- Secondary Metric: 30-day repeat purchase rate.

####    B. Product Level
- Hypothesis: Displaying a "Fast & Reliable Delivery" badge on products fulfilled by our most dependable logistics partners will increase the conversion rate for those products.
- Control Group (A): Product pages are displayed as they are now.
- Test Group (B): On eligible product pages, a visually appealing badge is shown near the "Add to Cart" button.
- Primary Metric: Add-to-cart rate and final conversion rate for badged products.

#### C. Marketing/Retention Level
- Hypothesis: A targeted win-back email campaign for churned customers who experienced a delivery delay is more effective than a generic win-back campaign.
- Control Group (A): Churned customers receive a standard "We Miss You" email with a 10% discount.
- Test Group (B): Churned customers who previously had a late order receive a targeted "We Messed Up. Give Us Another Chance" email that acknowledges the past issue and offers a more aggressive 25% discount.
- Primary Metric: Campaign conversion rate (number of churned customers who make a new purchase).