# NexGen Logistics – Predictive Delivery Optimizer

## How to Run

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Features

- **Delay Prediction**: Machine learning model predicts the likelihood of delivery delays
- **Route Risk Analysis**: Identifies high-risk routes and destinations
- **Cost Impact Visualization**: Shows how delays affect delivery costs
- **Customer Risk Dashboard**: Analyzes customer satisfaction in relation to delivery performance
- **Interactive Filters**: Filter data by priority, carrier, destination, and more
- **Actionable Recommendations**: Provides suggestions for high-risk orders

## Project Structure

```
nexgen-logistics/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── innovation_brief.pdf   # Business explanation (to be created)
└── data/                  # Directory containing CSV datasets
    ├── orders.csv
    ├── delivery_performance.csv
    ├── routes_distance.csv
    ├── cost_breakdown.csv
    ├── customer_feedback.csv
    ├── vehicle_fleet.csv
    └── warehouse_inventory.csv
```

## Data Sources

The application uses the following datasets:

1. **orders.csv**: Basic order information including priority, origin, and destination
2. **delivery_performance.csv**: Actual delivery dates and delay reasons
3. **routes_distance.csv**: Distance and travel time information for routes
4. **cost_breakdown.csv**: Detailed cost breakdown for each order
5. **customer_feedback.csv**: Customer ratings and feedback
6. **vehicle_fleet.csv**: Information about delivery vehicles
7. **warehouse_inventory.csv**: Warehouse capacity and utilization data

## Key Metrics

The application calculates several key metrics:

- **Delay Minutes**: Difference between actual and promised delivery times
- **Delay Flag**: Boolean indicator for delayed orders
- **Cost per Order**: Total cost of fulfilling each order
- **Delay Risk Score**: Probability of delay predicted by machine learning model
- **Customer Risk**: Combination of low rating and delivery delay

## Machine Learning Model

The application uses a Random Forest Classifier to predict delivery delays based on:

- Distance
- Estimated travel time
- Priority type
- Carrier
- Vehicle type

The model is trained on historical delivery data to identify patterns that lead to delays.