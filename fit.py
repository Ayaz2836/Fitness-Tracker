"""
Fitness Tracker Streamlit App
A comprehensive GUI for analyzing fitness data and predicting calories burned.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import kagglehub

# Set page configuration
st.set_page_config(
    page_title="Fitness Tracker App",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("💪 Fitness Tracker Analysis")
st.markdown("""
This app provides a comprehensive analysis of fitness tracking data including:
- **Data Overview**: Explore the dataset structure and statistics
- **Visualizations**: Interactive charts and correlation analysis
- **Machine Learning**: Predict calories burned using trained models
""")

# Load data function with caching
@st.cache_data
def load_data():
    """Load and return the fitness tracker dataset."""
    file_name = "gym_members_exercise_tracking_synthetic_data.csv"
    
    try:
        # Try downloading from kagglehub first
        path_full = kagglehub.dataset_download("nadeemajeedch/fitness-tracker-dataset")
        file_path = f"{path_full}/{file_name}"
        df = pd.read_csv(file_path)
    except Exception:
        # Fallback to local directory if Kaggle API fails
        try:
            df = pd.read_csv(file_name)
        except Exception:
            st.error(f"Could not load data. Please ensure '{file_name}' exists locally or your internet is connected.")
            return pd.DataFrame()
            
    return df

def preprocess_data(df):
    """Preprocess the data for analysis and modeling."""
    df = df.copy()
    
    # Clean 'Workout_Type' column
    df['Workout_Type'] = df['Workout_Type'].astype(str).str.strip()
    df['Workout_Type'] = df['Workout_Type'].str.replace('\\n', '', regex=False)
    df['Workout_Type'] = df['Workout_Type'].str.replace('\\t', '', regex=False)
    
    # Handle missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    
    for col in categorical_cols:
        if df[col].isnull().any() and len(df[col].mode()) > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def prepare_ml_data(df):
    """Prepare data for machine learning models."""
    df_ml = df.copy()
    
    # One-hot encoding for categorical features
    df_ml = pd.get_dummies(df_ml, columns=['Gender', 'Workout_Type'], drop_first=True)
    
    X = df_ml.drop(columns=['Calories_Burned'])
    y = df_ml['Calories_Burned']
    
    # Handle non-numeric columns
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(X.mean())
    
    return X, y

def train_models(X, y):
    """Train and return the ML models."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    return {
        'lr': {'model': lr_model, 'scaler': scaler, 'mse': mse_lr, 'r2': r2_lr},
        'rf': {'model': rf_model, 'scaler': scaler, 'mse': mse_rf, 'r2': r2_rf}
    }, X_train.columns

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Data Overview", "📈 Visualizations", "🤖 ML Prediction"]
)

# Load data
df = load_data()

if df.empty:
    st.error("No data loaded. Please check the dataset.")
    st.stop()

# Preprocess data
df = preprocess_data(df)

# Page 1: Data Overview
if page == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': [str(dtype) for dtype in df.dtypes.values],
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values
    })
    st.dataframe(col_info, use_container_width=True)
    
    st.subheader("Missing Values Details")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.dataframe(missing, use_container_width=True)
    else:
        st.success("✅ No missing values in the dataset!")

# Page 2: Visualizations
elif page == "📈 Visualizations":
    st.header("📈 Data Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization",
        ["Distribution of Calories Burned", "Correlation Heatmap", "Weight vs Calories", "Calories by Gender", "Workout Type Analysis"]
    )
    
    if viz_type == "Distribution of Calories Burned":
        st.subheader("Distribution of Calories Burned")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Calories_Burned'], bins=20, kde=True, ax=ax)
        ax.set_title('Distribution of Calories Burned')
        ax.set_xlabel('Calories Burned')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.75)
        st.pyplot(fig)
        
    elif viz_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        numerical_df = df.select_dtypes(include=[np.number])
        corr = numerical_df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Heatmap of Numerical Features')
        st.pyplot(fig)
        
    elif viz_type == "Weight vs Calories":
        st.subheader("Weight vs Calories Burned")
        gender_filter = st.multiselect("Select Gender", df['Gender'].unique(), default=df['Gender'].unique())
        df_filtered = df[df['Gender'].isin(gender_filter)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Weight (kg)', y='Calories_Burned', data=df_filtered, 
                     hue='Gender', palette='viridis', alpha=0.7, ax=ax)
        ax.set_title('Scatter Plot of Weight (kg) vs Calories Burned')
        ax.set_xlabel('Weight (kg)')
        ax.set_ylabel('Calories Burned')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        
    elif viz_type == "Calories by Gender":
        st.subheader("Calories Burned by Gender")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Gender', y='Calories_Burned', data=df, hue='Gender', 
                   palette='pastel', legend=False, ax=ax)
        ax.set_title('Distribution of Calories Burned by Gender')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Calories Burned')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
    elif viz_type == "Workout Type Analysis":
        st.subheader("Calories Burned by Workout Type")
        workout_types = df['Workout_Type'].unique()
        selected_workouts = st.multiselect("Select Workout Types", workout_types, default=workout_types[:5])
        df_workout = df[df['Workout_Type'].isin(selected_workouts)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='Workout_Type', y='Calories_Burned', data=df_workout, ax=ax)
        ax.set_title('Calories Burned by Workout Type')
        ax.set_xlabel('Workout Type')
        ax.set_ylabel('Calories Burned')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

# Page 3: ML Prediction
elif page == "🤖 ML Prediction":
    st.header("🤖 ML Prediction")
    st.markdown("Predict calories burned using trained machine learning models.")
    
    # Train models
    X, y = prepare_ml_data(df)
    models, feature_names = train_models(X, y)
    
    # Model selection
    model_type = st.radio("Select Model", ["Linear Regression", "Random Forest"])
    
    if model_type == "Linear Regression":
        selected_model = models['lr']
        st.info(f"📊 Linear Regression - MSE: {selected_model['mse']:.2f}, R²: {selected_model['r2']:.2f}")
    else:
        selected_model = models['rf']
        st.info(f"🌲 Random Forest - MSE: {selected_model['mse']:.2f}, R²: {selected_model['r2']:.2f}")
    
    st.markdown("---")
    st.subheader("Enter Your Details")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        duration = st.number_input("Workout Duration (minutes)", min_value=10.0, max_value=180.0, value=45.0)
    
    with col2:
        avg_bpm = st.number_input("Average BPM", min_value=60, max_value=200, value=120)
        max_bpm = st.number_input("Max BPM", min_value=80, max_value=220, value=160)
        resting_bpm = st.number_input("Resting BPM", min_value=40, max_value=100, value=60)
    
    # Dropdown inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    workout_type = st.selectbox("Workout Type", df['Workout_Type'].unique())
    
    # Create prediction
    if st.button("🔮 Predict Calories Burned"):
        # Create input dataframe matching feature structure
        input_data = {}
        
        for col in feature_names:
            if col == 'Age':
                input_data[col] = age
            elif col == 'Weight (kg)':
                input_data[col] = weight
            elif col == 'Height (cm)':
                input_data[col] = height
            elif col == 'Duration':
                input_data[col] = duration
            elif col == 'Avg_BPM':
                input_data[col] = avg_bpm
            elif col == 'Max_BPM':
                input_data[col] = max_bpm
            elif col == 'Resting_BPM':
                input_data[col] = resting_bpm
            elif col == 'Gender_Male':
                input_data[col] = 1 if gender == 'Male' else 0
            elif col.startswith('Workout_Type_'):
                workout_type_col = col.replace('Workout_Type_', '')
                input_data[col] = 1 if workout_type == workout_type_col else 0
            else:
                input_data[col] = 0
        
        # Create DataFrame and predict
        input_df = pd.DataFrame([input_data])
        
        # Ensure column order matches training data
        input_df = input_df[feature_names]
        
        # Scale and predict
        input_scaled = selected_model['scaler'].transform(input_df)
        prediction = selected_model['model'].predict(input_scaled)[0]
        
        st.success(f"🔥 Predicted Calories Burned: **{prediction:.2f}**")
        
        # Display input summary
        with st.expander("View Input Summary"):
            st.json({
                'Age': age,
                'Weight (kg)': weight,
                'Height (cm)': height,
                'Duration (minutes)': duration,
                'Average BPM': avg_bpm,
                'Max BPM': max_bpm,
                'Resting BPM': resting_bpm,
                'Gender': gender,
                'Workout Type': workout_type
            })

    # Display feature importance for Random Forest
    if model_type == "Random Forest":
        st.markdown("---")
        st.subheader("Feature Importance")
        
        rf_model = models['rf']['model']
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
        ax.set_title('Top 10 Feature Importances (Random Forest)')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("💪 Fitness Tracker App v1.0")
st.sidebar.markdown("Built with Streamlit")