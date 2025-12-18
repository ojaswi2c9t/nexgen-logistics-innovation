import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from typing import Tuple, Optional, Any

# Set page config
st.set_page_config(
    page_title="NexGen Predictive Delivery Optimizer",
    page_icon="🚚",
    layout="wide"
)

# Title
st.title("🚚 NexGen Predictive Delivery Optimizer")
st.markdown("##### Case Study- LOGISTICS INNOVATION CHALLENGE")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📦 Overview",
    "🚨 Delay Risk Predictor",
    "🛣 Route Analysis",
    "💰 Cost Impact",
    "⭐ Customer Experience"
])

# Create delay risk prediction model
# This function creates a model to predict delivery delays
@st.cache_resource
def create_delay_model(df):
    # Prepare features for modeling
    features = ['distance_km', 'estimated_travel_time_hours', 'priority', 'carrier', 'vehicle_type']
    target = 'is_delayed'
    
    # Select relevant columns
    model_df = df[features + [target]].copy()
    
    # Encode categorical variables
    le_priority = LabelEncoder()
    le_carrier = LabelEncoder()
    le_vehicle = LabelEncoder()
    
    model_df['priority_encoded'] = le_priority.fit_transform(model_df['priority'])
    model_df['carrier_encoded'] = le_carrier.fit_transform(model_df['carrier'])
    model_df['vehicle_type_encoded'] = le_vehicle.fit_transform(model_df['vehicle_type'])
    
    # Select final features for model
    X = model_df[['distance_km', 'estimated_travel_time_hours', 'priority_encoded', 'carrier_encoded', 'vehicle_type_encoded']]
    y = model_df[target].astype(int)
    
    # Handle missing values
    X = X.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, le_priority, le_carrier, le_vehicle

# Load data function
@st.cache_data
def load_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    # Check if data directory exists
    if not os.path.exists('data'):
        st.error("Data directory not found. Please ensure the data folder with CSV files exists.")
        return None, None
    
    try:
        orders = pd.read_csv('data/orders.csv')
        delivery = pd.read_csv('data/delivery_performance.csv')
        routes = pd.read_csv('data/routes_distance.csv')
        costs = pd.read_csv('data/cost_breakdown.csv')
        feedback = pd.read_csv('data/customer_feedback.csv')
        vehicles = pd.read_csv('data/vehicle_fleet.csv')
        warehouses = pd.read_csv('data/warehouse_inventory.csv')
        
        # Merge datasets
        df = orders.merge(delivery, on='order_id', how='left')
        df = df.merge(routes, on=['origin_warehouse', 'destination_city'], how='left')
        df = df.merge(costs, on='order_id', how='left')
        df = df.merge(feedback, on='order_id', how='left')
        df = df.merge(vehicles, left_on='carrier', right_on='carrier', how='left')
        
        # Convert date columns
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['promised_delivery_date'] = pd.to_datetime(df['promised_delivery_date'])
        df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'])
        
        # Create derived metrics
        df['delay_minutes'] = (df['actual_delivery_date'] - df['promised_delivery_date']).dt.total_seconds() / 60
        df['is_delayed'] = df['delay_minutes'] > 0
        df['delay_hours'] = df['delay_minutes'] / 60
        df['cost_per_order'] = df['total_cost']
        df['delivery_speed'] = df['distance_km'] / df['estimated_travel_time_hours']
        
        # Handle negative delays (early deliveries)
        df['delay_minutes'] = df['delay_minutes'].fillna(0)
        df['delay_hours'] = df['delay_hours'].fillna(0)
        
        return df, warehouses
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Load data
data_result = load_data()
if data_result is not None and data_result[0] is not None and data_result[1] is not None:
    df, warehouses = data_result
    # Check if df is valid before proceeding
    if df is None or df.empty:
        st.error("No data available to display.")
        st.stop()
else:
    st.error("Failed to load data. Please check the data files and try again.")
    st.stop()

# Create model only if df is valid
try:
    model, le_priority, le_carrier, le_vehicle = create_delay_model(df)
    
    # Add delay risk predictions to dataframe
    def predict_delay_risk(row):
        try:
            priority_encoded = le_priority.transform([row['priority']])[0] if row['priority'] in le_priority.classes_ else 0
            carrier_encoded = le_carrier.transform([row['carrier']])[0] if row['carrier'] in le_carrier.classes_ else 0
            vehicle_type_encoded = le_vehicle.transform([row['vehicle_type']])[0] if row['vehicle_type'] in le_vehicle.classes_ else 0
            
            features = np.array([[row['distance_km'], row['estimated_travel_time_hours'], 
                                 priority_encoded, carrier_encoded, vehicle_type_encoded]])
            risk_score = model.predict_proba(features)[0][1]
            return risk_score
        except:
            return 0.5

    df['delay_risk'] = df.apply(predict_delay_risk, axis=1)
except Exception as e:
    st.error(f"Error creating prediction model: {str(e)}")
    st.stop()

# Page routing
if page == "📦 Overview":
    st.header("📦 Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_orders = len(df)
    delayed_orders = df['is_delayed'].sum()
    delay_percentage = (delayed_orders / total_orders) * 100 if total_orders > 0 else 0
    avg_delay = df[df['is_delayed']]['delay_hours'].mean()
    
    col1.metric("Total Orders", total_orders)
    col2.metric("Delayed Orders", delayed_orders)
    col3.metric("Delay Rate", f"{delay_percentage:.1f}%")
    col4.metric("Avg Delay (Hours)", f"{avg_delay:.1f}" if not np.isnan(avg_delay) else "0.0")
    
    # Delay trend chart
    st.subheader("Delivery Performance Trend")
    df['month'] = df['order_date'].dt.to_period('M').astype(str)
    monthly_delays = df.groupby('month')['is_delayed'].mean().reset_index()
    monthly_delays['delay_percentage'] = monthly_delays['is_delayed'] * 100
    
    fig = px.line(monthly_delays, x='month', y='delay_percentage', 
                  title='Monthly Delay Percentage',
                  labels={'delay_percentage': 'Delay Percentage (%)', 'month': 'Month'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Delay reasons
    st.subheader("Delay Reasons")
    delayed_df = df[df['is_delayed']]
    if 'delay_reason' in delayed_df.columns:
        # Convert to pandas Series to ensure proper type
        delay_reason_series = pd.Series(delayed_df['delay_reason'])
        delay_reasons = delay_reason_series.value_counts()
        if len(delay_reasons) > 0:
            # Extract values and index as lists
            reasons = list(delay_reasons.index)
            counts = list(delay_reasons.values)
            fig = px.bar(x=reasons, y=counts,
                         labels={'x': 'Reason', 'y': 'Number of Delays'},
                         title='Causes of Delivery Delays')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No delay reasons recorded.")
    else:
        st.info("Delay reason data not available.")

elif page == "🚨 Delay Risk Predictor":
    st.header("🚨 Delay Risk Predictor")
    
    # Filters
    st.sidebar.subheader("Filters")
    priority_filter = st.sidebar.multiselect("Priority", df['priority'].unique(), df['priority'].unique())
    carrier_filter = st.sidebar.multiselect("Carrier", df['carrier'].unique(), df['carrier'].unique())
    
    # Apply filters
    filtered_df = df[
        (df['priority'].isin(priority_filter)) &
        (df['carrier'].isin(carrier_filter))
    ]
    
    # Risk distribution
    st.subheader("Delay Risk Distribution")
    fig = px.histogram(filtered_df, x='delay_risk', nbins=20,
                       title='Distribution of Delay Risk Scores',
                       labels={'delay_risk': 'Delay Risk Probability'})
    st.plotly_chart(fig, use_container_width=True)
    
    # High-risk orders
    st.subheader("High-Risk Orders")
    risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7, 0.05)

    # Manual filtering to avoid pandas operations that cause type issues
    high_risk_data = []
    for idx in range(len(filtered_df)):
        row = filtered_df.iloc[idx]
        if row['delay_risk'] >= risk_threshold:
            high_risk_data.append({
                'order_id': row['order_id'],
                'destination_city': row['destination_city'],
                'priority': row['priority'],
                'carrier': row['carrier'],
                'delay_risk': row['delay_risk']
            })

    if len(high_risk_data) > 0:
        st.write(f"Found {len(high_risk_data)} high-risk orders:")
        # Sort manually by delay_risk
        sorted_data = sorted(high_risk_data, key=lambda x: x['delay_risk'], reverse=True)
        # Display as dataframe
        st.dataframe(pd.DataFrame(sorted_data))
        
        # Recommendations
        st.subheader("Recommended Actions")
        st.info("For high-risk orders, consider:")
        st.markdown("""
        - Changing carrier to a more reliable one
        - Upgrading to a faster vehicle type
        - Rerouting via low-traffic paths
        - Switching to express delivery
        """)
    else:
        st.success("No high-risk orders found with current filters.")

    # Download button for high-risk orders
    if len(high_risk_data) > 0:
        # Convert to CSV
        csv_buffer = pd.DataFrame(high_risk_data).to_csv(index=False)
        st.download_button(
            label="Download High-Risk Orders",
            data=csv_buffer,
            file_name="high_risk_orders.csv",
            mime="text/csv"
        )

elif page == "🛣 Route Analysis":
    st.header("🛣 Route Analysis")
    
    # Filters
    st.sidebar.subheader("Filters")
    route_filter = st.sidebar.multiselect("Destination City", df['destination_city'].unique(), df['destination_city'].unique())
    
    # Apply filters
    filtered_df = df[df['destination_city'].isin(route_filter)]
    
    # Delay by destination
    st.subheader("Delay Percentage by Destination")
    dest_delay = filtered_df.groupby('destination_city')['is_delayed'].mean().reset_index()
    dest_delay['delay_percentage'] = dest_delay['is_delayed'] * 100
    dest_delay = dest_delay.sort_values('delay_percentage', ascending=False)
    
    fig = px.bar(dest_delay, x='destination_city', y='delay_percentage',
                 labels={'delay_percentage': 'Delay Percentage (%)', 'destination_city': 'Destination'},
                 title='Delay Percentage by Destination City')
    st.plotly_chart(fig, use_container_width=True)
    
    # Distance vs Delay
    st.subheader("Distance vs Delay")
    fig = px.scatter(filtered_df, x='distance_km', y='delay_hours',
                     color='is_delayed', hover_data=['order_id'],
                     labels={'distance_km': 'Distance (km)', 'delay_hours': 'Delay (hours)'},
                     title='Distance vs Delay Hours')
    st.plotly_chart(fig, use_container_width=True)
    
    # Carrier performance
    st.subheader("Carrier Performance")
    carrier_stats = filtered_df.groupby('carrier').agg({
        'is_delayed': 'mean',
        'delay_hours': 'mean',
        'order_id': 'count'
    }).reset_index()
    carrier_stats.columns = ['Carrier', 'Delay Rate', 'Avg Delay Hours', 'Order Count']
    carrier_stats['Delay Rate'] = carrier_stats['Delay Rate'] * 100
    
    fig = px.bar(carrier_stats, x='Carrier', y='Delay Rate',
                 hover_data=['Avg Delay Hours', 'Order Count'],
                 labels={'Delay Rate': 'Delay Rate (%)'},
                 title='Carrier Delay Rates')
    st.plotly_chart(fig, use_container_width=True)

elif page == "💰 Cost Impact":
    st.header("💰 Cost Impact Analysis")
    
    # Filters
    st.sidebar.subheader("Filters")
    priority_filter = st.sidebar.multiselect("Priority", df['priority'].unique(), df['priority'].unique())
    
    # Apply filters
    filtered_df = df[df['priority'].isin(priority_filter)]
    
    # Cost vs Delay
    st.subheader("Cost Impact of Delays")
    fig = px.scatter(filtered_df, x='delay_hours', y='total_cost',
                     color='is_delayed', hover_data=['order_id'],
                     labels={'delay_hours': 'Delay (hours)', 'total_cost': 'Total Cost ($)'},
                     title='Delay Hours vs Total Cost')
    st.plotly_chart(fig, use_container_width=True)
    
    # Average cost by delay status
    st.subheader("Average Cost by Delay Status")
    cost_by_delay = filtered_df.groupby('is_delayed')['total_cost'].mean().reset_index()

    # Manually create the data for plotting to avoid type issues
    x_data = []
    y_data = []

    # Iterate through rows using iloc to avoid type issues
    for i in range(len(cost_by_delay)):
        is_delayed_val = cost_by_delay.iloc[i]['is_delayed']
        avg_cost_val = cost_by_delay.iloc[i]['total_cost']
        status_label = 'Delayed' if is_delayed_val else 'On Time'
        x_data.append(status_label)
        y_data.append(avg_cost_val)

    fig = px.bar(x=x_data, y=y_data,
                 labels={'x': 'Status', 'y': 'Average Cost ($)'},
                 title='Average Cost: Delayed vs On-Time Deliveries')
    st.plotly_chart(fig, use_container_width=True)

    # Cost breakdown
    st.subheader("Cost Breakdown Analysis")
    cost_cols = ['fuel_cost', 'labor_cost', 'maintenance_cost', 'insurance_cost']

    # Calculate mean values manually to avoid type issues
    cost_values = []
    for col in cost_cols:
        # Calculate mean manually
        col_sum = filtered_df[col].sum()
        col_count = len(filtered_df[col])
        mean_val = col_sum / col_count if col_count > 0 else 0
        cost_values.append(mean_val)

    fig = px.pie(values=cost_values, names=cost_cols,
                 title='Average Cost Breakdown per Order')
    st.plotly_chart(fig, use_container_width=True)

# Customer Experience section
elif page == "⭐ Customer Experience":
    st.header("⭐ Customer Experience")
    
    # Filters
    st.sidebar.subheader("Filters")
    rating_filter = st.sidebar.slider("Minimum Rating", 1, 5, 1)
    
    # Apply filters
    filtered_df = df[df['customer_rating'] >= rating_filter]
    
    # Rating distribution
    st.subheader("Customer Rating Distribution")
    
    # Manually calculate value counts to avoid type issues
    rating_counts = {}
    for rating in filtered_df['customer_rating']:
        if rating in rating_counts:
            rating_counts[rating] += 1
        else:
            rating_counts[rating] = 1
    
    # Sort the ratings
    sorted_ratings = sorted(rating_counts.keys())
    rating_values = [rating_counts[rating] for rating in sorted_ratings]
    
    fig = px.bar(x=sorted_ratings, y=rating_values,
                 labels={'x': 'Rating', 'y': 'Number of Orders'},
                 title='Customer Rating Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Rating vs Delay
    st.subheader("Rating vs Delay")
    # Group by customer rating and calculate mean delay
    rating_delay_data = {}
    # Get unique ratings manually
    unique_ratings = []
    for rating in filtered_df['customer_rating']:
        if rating not in unique_ratings:
            unique_ratings.append(rating)

    for rating in unique_ratings:
        subset = filtered_df[filtered_df['customer_rating'] == rating]
        mean_delay = subset['delay_hours'].mean()
        rating_delay_data[rating] = mean_delay
    
    # Sort by rating
    sorted_items = sorted(rating_delay_data.items())
    ratings = [item[0] for item in sorted_items]
    delays = [item[1] for item in sorted_items]
    
    fig = px.line(x=ratings, y=delays,
                  markers=True,
                  labels={'x': 'Customer Rating', 'y': 'Average Delay (hours)'},
                  title='Average Delay by Customer Rating')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feedback analysis
    st.subheader("Feedback Analysis")
    if 'delivery_satisfaction' in filtered_df.columns:
        # Manually count satisfaction levels
        satisfaction_counts = {}
        for satisfaction in filtered_df['delivery_satisfaction']:
            if pd.notna(satisfaction):  # Check for NaN values
                if satisfaction in satisfaction_counts:
                    satisfaction_counts[satisfaction] += 1
                else:
                    satisfaction_counts[satisfaction] = 1
        
        if satisfaction_counts:
            satisfaction_labels = list(satisfaction_counts.keys())
            satisfaction_values = list(satisfaction_counts.values())
            
            fig = px.pie(values=satisfaction_values, names=satisfaction_labels,
                         title='Delivery Satisfaction Levels')
            st.plotly_chart(fig, use_container_width=True)
    
    # Low rating analysis
    st.subheader("Low Rating Analysis")
    low_ratings = df[df['customer_rating'] <= 2]
    if len(low_ratings) > 0:
        st.write(f"Found {len(low_ratings)} orders with low ratings (≤2):")
        st.dataframe(low_ratings[['order_id', 'customer_rating', 'delay_hours', 'feedback_comments']])
    else:
        st.success("No low-rated orders found.")

# Footer
st.markdown("---")
st.markdown("🚚 NexGen Predictive Delivery Optimizer | Helping you optimize logistics and reduce delivery delays")
st.markdown("Developed by : Ojaswi Gahoi - 22BCE10783, ojaswigahoi@gmail.com, https://github.com/ojaswi2c9t")
