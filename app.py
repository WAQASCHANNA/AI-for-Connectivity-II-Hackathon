import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from prophet import Prophet

# ----------------------
# Data Loading & Processing
# ----------------------
@st.cache_data
def load_synthetic_data():
    """Generate comprehensive synthetic dataset"""
    # School/healthcare facility data
    facilities = pd.DataFrame({
        "facility_id": range(1, 51),
        "latitude": np.random.uniform(35, 45, 50),
        "longitude": np.random.uniform(-100, -80, 50),
        "type": np.random.choice(["School", "Clinic"], 50),
        "connectivity_score": np.random.randint(20, 95, 50)
    })
    
    # Equipment sensor data
    equipment = pd.DataFrame({
        "facility_id": np.repeat(range(1, 51), 5),
        "timestamp": pd.date_range("2024-01-01", periods=250, freq="H"),
        "temperature": np.random.normal(40, 5, 250),
        "vibration": np.random.gamma(1, 2, 250),
        "power_usage": np.random.normal(200, 50, 250)
    })
    
    return facilities, equipment

# ----------------------
# AI Models
# ----------------------
def train_energy_model(X, y):
    model = RandomForestRegressor(n_estimators=20)
    model.fit(X, y)
    return model

def train_maintenance_model(df):
    model = IsolationForest(contamination=0.1)
    features = ["temperature", "vibration", "power_usage"]
    model.fit(df[features])
    return model

# ----------------------
# Streamlit App
# ----------------------
st.set_page_config(page_title="Public Sector Network Optimizer", layout="wide")
st.title("🏥 AI-Powered Network Management for Public Institutions")

# Load data
facilities, equipment = load_synthetic_data()

# ----------------------
# Enhanced Visualizations
# ----------------------
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Facility Map", "📈 Trends", "🔧 Maintenance", "💰 Cost Analysis"])

with tab1:
    st.subheader("Facility Network Health")
    
    # Merge facility and equipment data
    merged_df = facilities.merge(
        equipment.groupby("facility_id")["power_usage"].mean().reset_index(),
        on="facility_id"
    )
    
    # Add AI recommendations
    merged_df["recommendation"] = np.where(
        merged_df["power_usage"] > 200,
        "🛠️ Adjust router sleep cycles",
        "✅ Stable configuration"
    )
    
    # Create visualization parameters
    merged_df["size"] = np.interp(merged_df["power_usage"], [150, 250], [10, 30])
    merged_df["status"] = np.where(merged_df["connectivity_score"] < 50, "At Risk", "Stable")
    
    fig = px.scatter_mapbox(
        merged_df,
        lat="latitude",
        lon="longitude",
        color="status",
        size="size",
        hover_name="type",
        hover_data=["connectivity_score", "power_usage", "recommendation"],
        mapbox_style="carto-positron",
        zoom=4,
        color_discrete_map={"At Risk": "#e74c3c", "Stable": "#2ecc71"}
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    csv = merged_df[["facility_id", "type", "power_usage", "recommendation"]].to_csv(index=False)
    st.download_button(
        "📥 Export Facility Report",
        data=csv,
        file_name="gridguardians_report.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Energy Consumption Forecast")
    
    selected_facility = st.selectbox("Select Facility", facilities["facility_id"])
    facility_data = equipment[equipment["facility_id"] == selected_facility]
    
    # Enhanced 72-hour forecast
    try:
        prophet_df = facility_data.rename(columns={"timestamp": "ds", "power_usage": "y"})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=72, freq="H")  # 72-hour forecast
        forecast = model.predict(future)
        
        fig = px.line(forecast, x="ds", y="yhat", 
                     title="72-Hour Power Usage Forecast",
                     labels={"ds": "Time", "yhat": "Predicted Power Usage (kWh)"})
        fig.add_scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Actual Usage")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Forecasting error: {str(e)}")

with tab3:
    st.subheader("Equipment Health Monitoring")
    
    # Train anomaly detection model
    model = train_maintenance_model(equipment)
    equipment["anomaly_score"] = model.decision_function(equipment[["temperature", "vibration", "power_usage"]])
    equipment["needs_maintenance"] = equipment["anomaly_score"] < np.percentile(equipment["anomaly_score"], 10)
    
    # Display critical alerts with priority
    critical_issues = equipment[equipment["needs_maintenance"]].merge(facilities, on="facility_id")
    if not critical_issues.empty:
        # Calculate priority scores
        critical_issues["priority"] = np.interp(
            critical_issues["anomaly_score"],
            [critical_issues["anomaly_score"].min(), critical_issues["anomaly_score"].max()],
            [1, 10]
        ).astype(int)
        
        st.write("🚨 Critical Maintenance Needed")
        st.dataframe(
            critical_issues.sort_values("priority", ascending=False)[["facility_id", "type", "priority", "timestamp", "temperature"]],
            hide_index=True,
            column_config={
                "timestamp": "Last Reading",
                "temperature": st.column_config.ProgressColumn(
                    "Temperature (°C)",
                    help="Equipment temperature",
                    format="%.1f°C",
                    min_value=30,
                    max_value=60
                )
            }
        )
    else:
        st.success("✅ All equipment operating normally")

with tab4:
    st.subheader("Financial & Environmental Impact")
    
    col1, col2 = st.columns(2)
    with col1:
        energy_cost = st.slider("Energy Cost ($/kWh)", 0.10, 1.00, 0.25)
        labor_cost = st.slider("Hourly Labor Cost ($)", 20, 100, 45)
    with col2:
        maint_duration = st.slider("Maintenance Duration (hours)", 1, 8, 2)
        carbon_price = st.slider("Carbon Price ($/ton)", 10, 100, 50)
    
    # Calculate savings
    total_power = equipment["power_usage"].sum()
    predicted_power = total_power * 0.85  # Assume 15% reduction
    savings = (total_power - predicted_power) * energy_cost
    
    # Maintenance costs
    num_issues = len(critical_issues) if 'critical_issues' in locals() else 0
    maint_cost = num_issues * labor_cost * maint_duration
    
    # Environmental impact
    co2_reduction = (total_power - predicted_power) * 0.5  # kg CO2
    carbon_credits = (co2_reduction / 1000) * carbon_price  # tons
    
    # Display metrics
    st.metric("Monthly Energy Savings Potential", f"${savings:,.2f}")
    st.metric("Maintenance Cost Estimate", f"${maint_cost:,.2f}")
    st.metric("CO₂ Reduction Impact", f"{co2_reduction:,.1f} kg (${carbon_credits:,.2f} credits)")
    
    # Cost-benefit visualization
    cost_data = pd.DataFrame({
        "Category": ["Savings", "Maintenance"],
        "Amount": [savings, -maint_cost]
    })
    fig = px.bar(cost_data, x="Category", y="Amount", 
                title="Net Financial Impact",
                color="Category",
                color_discrete_map={"Savings": "#2ecc71", "Maintenance": "#e74c3c"})
    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# About Section
# ----------------------
st.sidebar.markdown("""
**Key Features**
- Real-time facility monitoring
- Predictive maintenance alerts
- Energy cost forecasting
- Sustainability impact analysis

**Next Steps**
- Integrate IoT sensor data
- Add multi-language support
- Implement API-based alerting
""")
