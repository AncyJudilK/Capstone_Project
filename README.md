# Electric Vehicle Forecasting Dashboard

## Overview
This project focuses on analyzing and forecasting trends in the electric vehicle (EV) population using machine learning algorithms. It provides an intuitive interface through Streamlit, enabling users to upload custom datasets or utilize a sample dataset. Key features include:

- **Exploratory Data Analysis (EDA)**
- **Feature Engineering**
- **Model Training and Evaluation**
- **Interactive Visualization**

The Gradient Boosting Regressor is employed to predict the electric range of EVs based on multiple factors.

---

## Scope of the Project
With the global shift toward sustainability, the adoption of electric vehicles has been growing rapidly. This project aims to provide actionable insights for policymakers, manufacturers, and consumers by:

- **Analyzing trends** in the EV population, including state-wise distribution and vehicle performance metrics.
- **Forecasting critical parameters**, such as the electric range using advanced machine learning models.
- **Enhancing decision-making** by identifying key factors influencing EV performance.

---

## Dataset Description
The dataset contains detailed records of electric vehicles, including their model year, make, type, range, price, and geographical distribution. It comprises a mix of numerical and categorical features, offering insights into EV trends and performance.

### Data Path
`Electric_Vehicle_Population_Data.xlsx`

### Required Columns
- **Model Year**: Year of manufacture.
- **Make**: Manufacturer of the EV.
- **Model**: Specific model name.
- **Electric Vehicle Type**: Type of EV (e.g., Battery Electric Vehicle).
- **Electric Range**: Maximum distance on a full charge (in miles).
- **Base MSRP**: Manufacturer's Suggested Retail Price.
- **County**: Registration location (county).
- **State**: Registration location (state).

---

## Exploratory Data Analysis (EDA)
### Visualizations and Insights
- **Distribution of EVs by State**: A bar chart visualizing the count of EVs across different states.
- **Average Electric Range by Make**: A line chart showcasing manufacturers' average EV ranges.

---

## Feature Engineering
To enhance the dataset's usability, the following features were derived:
- **Vehicle Age**: Calculated as the difference between the current year and the model year.
- **Price per Mile**: Calculated as `Base MSRP / Electric Range`.

Missing or infinite values were handled using imputation and replacement techniques.

---

## Model Training and Forecasting
### Target Variable
**Electric Range**

### Features Used
- Vehicle Age
- Price per Mile
- Base MSRP

### Model Used
**Gradient Boosting Regressor**
Selected for its robustness in handling tabular datasets with numerical and categorical features.

### Model Training Process
1. **Data Split**: 70% Training, 30% Testing.
2. **Hyperparameter Tuning**: Performed using `GridSearchCV` to optimize parameters like learning rate, number of estimators, and maximum depth.

### Metrics Evaluated
- **Mean Absolute Error (MAE)**: Measures average prediction errors.
- **Root Mean Squared Error (RMSE)**: Penalizes large errors more heavily.
- **R² Score**: Indicates the proportion of variance explained by the model.

---

## Visualization of Results
- **Actual vs. Predicted Values**: Tabular comparison for quick insights.
- **Interactive Line Chart**: Visual comparison of actual and predicted distributions.

---

## Challenges
1. **Data Quality**: Missing or inconsistent data required extensive cleaning and imputation.
2. **Feature Selection**: Identifying impactful features for prediction while avoiding multicollinearity.
3. **Model Optimization**: Fine-tuning the Gradient Boosting Regressor for optimal performance required extensive computation and experimentation.

---

## Running the Application
### Using Streamlit Interface
```bash
streamlit run ev.py
```

### Using API Integration
1. Start the API server:
   ```bash
   python app.py
   ```
2. Run the Streamlit dashboard:
   ```bash
   streamlit run dashboard.py
   ```

---

## GitHub Repository
Explore the project on GitHub: [Electric Vehicle Forecasting Dashboard](https://github.com/AncyJudilK/Capstone_Project/tree/ccb10e35778157d076ef8f931492d08baec1fc1c/Electric_Vehicle_Population)

---
