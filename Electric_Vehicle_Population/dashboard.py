import pandas as pd
import numpy as np
import streamlit as st
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import altair as alt
import time
import os
import plotly.express as px
import plotly.graph_objects as go

# Streamlit Page Configuration
st.set_page_config(page_title="Electric Vehicle Dashboard", page_icon="🚗", layout="wide")

# Title and Description
st.title("🚗 Electric Vehicle Analysis Dashboard")
st.write("Analyze and forecast trends in electric vehicle data.")

# Fetch Data from Flask API
@st.cache_data
def fetch_ev_data():
    try:
        response = requests.get('http://127.0.0.1:5000/get_ev_data', timeout=5)  # Set a timeout
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        return pd.DataFrame(response.json())
    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to the Flask API. Ensure the API is running and accessible.")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("The request to the API timed out. Please try again later.")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching data: {e}")
        st.stop()

# Load Data
with st.spinner("Fetching data from the API..."):
    data = fetch_ev_data()
st.success("Data fetched successfully!")
st.dataframe(data.head(10))

# Exploratory Data Analysis (EDA) Section
st.header("📊 Exploratory Data Analysis")
st.write("Analyze trends and insights from the dataset.")

# Distribution of EVs by State
try:
    state_count = data['State'].value_counts().reset_index()
    state_count.columns = ['State', 'Count']
    st.subheader("Electric Vehicle Distribution by State")
    bar_chart = alt.Chart(state_count).mark_bar().encode(
        x='State',
        y='Count',
        tooltip=['State', 'Count']
    ).properties(width=700, height=400)
    st.altair_chart(bar_chart, use_container_width=True)
except KeyError as e:
    st.error(f"EDA Error: {e}")
    st.stop()

# Average Electric Range by Make
try:
    range_by_make = data.groupby('Make')['Electric Range'].mean().reset_index()
    range_by_make = range_by_make.sort_values('Electric Range', ascending=False)
    st.subheader("Average Electric Range by Vehicle Make")
    line_chart = alt.Chart(range_by_make).mark_line(point=True).encode(
        x='Make',
        y='Electric Range',
        tooltip=['Make', 'Electric Range']
    ).properties(width=700, height=400)
    st.altair_chart(line_chart, use_container_width=True)
except KeyError as e:
    st.error(f"EDA Error: {e}")
    st.stop()

# Feature Engineering Section
st.header("🔧 Feature Engineering")
st.write("Generate new features to enhance the dataset.")

# Feature Engineering
try:
    current_year = pd.Timestamp.now().year
    data['Vehicle Age'] = current_year - data['Model Year']
    data['Price per Mile'] = data['Base MSRP'] / data['Electric Range']
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

    st.write("Enhanced Dataset:")
    st.dataframe(data.head())
except KeyError as e:
    st.error(f"Feature Engineering Error: {e}")
    st.stop()

# Forecasting Section
st.header("📈 Forecasting")
st.write("Use machine learning to predict the electric range of electric vehicles.")

# Define Features and Target Variable
features = ['Vehicle Age', 'Price per Mile', 'Base MSRP']
target_variable = 'Electric Range'

# Display Target Variable
st.write(f"Target Variable: **{target_variable}**")

# Data Preparation
X = data[features]
y = data[target_variable]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dropdown for target variable selection
target_variable = st.selectbox("Select Target Variable", ["Electric Range"],
                               format_func=lambda x: 'Select a Target Variable'
                               if x == "" else x)

st.markdown("""<h6 style="color: #1ABC9C;"> Gradient Boosting is the best choice for forecasting </h6>""",
            unsafe_allow_html=True)

# Model Training and Forecasting
if st.button("Run Forecasting"):
    if target_variable == "":
        st.error("Please select a target variable before running the forecasting.")
    else:
        st.write(f"Running the **Gradient Boosting** model with target variable **{target_variable}**...")
        try:
            with st.spinner('🔄 Performing hyperparameter tuning and training the model...'):
                time.sleep(3)  # Simulate loading time
                # Hyperparameter tuning with GridSearchCV
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                }
            with st.spinner('🔄 Making predictions with the best model...'):
                time.sleep(3)  # Simulate prediction time
                # Make predictions with the best model    
                model = GradientBoostingRegressor(random_state=42)
                grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_

                # Predictions
                y_pred = best_model.predict(X_test)

                # Metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = mean_absolute_percentage_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.success("Forecasting completed!")
                st.subheader("Model Performance Metrics")
                st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
                st.metric("Root Mean Square Error (RMSE)", f"{rmse:.2f}")
                st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2%}")
                st.metric("R² Score", f"{r2:.2f}")

                st.write("### Best Model Hyperparameters")
                st.write(grid_search.best_params_)

                # Display Actual vs Predicted Values
                forecast_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
                st.write("### Actual vs Predicted Values")
                st.dataframe(forecast_results.head(10))
        except Exception as e:
                st.error(f"Model Training Error: {e}")

            # Add an Actual vs Predicted Line Chart with Interactive Features
        
    
        # Line Chart: Actual vs Predicted (For First 50 Data Points)
        try:
            st.write("### Line Chart: Actual vs Predicted Over Data Index")
        
            # Select first 50 data points
            y_test_subset = y_test[:50]
            y_pred_subset = y_pred[:50]
        
            line_fig = go.Figure()
            line_fig.add_trace(
                go.Scatter(
                    y=y_test_subset,
                    mode='lines',
                    name="Actual Values",
                    line=dict(color="blue")
                )
            )
            line_fig.add_trace(
                go.Scatter(
                    y=y_pred_subset,
                    mode='lines',
                    name="Predicted Values",
                    line=dict(color="orange", dash="dash")
                )
            )
            line_fig.update_layout(
                title="Actual vs Predicted Line Chart ",
                xaxis_title="Data Index",
                yaxis_title="Values",
                width=800,
                height=500
            )
            st.plotly_chart(line_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in generating the line chart: {e}")
