# Fitness-Tracker
💪 Fitness Tracker Analysis App
A comprehensive, interactive web application built with Streamlit for analyzing gym exercise data and predicting calories burned using Machine Learning.

🚀 Overview
This project provides a complete data science workflow—from automated data ingestion and preprocessing to interactive visualization and predictive modeling. It allows users to explore fitness trends and estimate energy expenditure based on physiological and workout parameters.

✨ Features
📊 Data Overview: Interactive exploration of the dataset, including statistical summaries, data types, and health-check metrics for missing values.

📈 Visualizations:

Distribution analysis of calories burned.

Correlation heatmaps for numerical features.

Comparative scatter plots (Weight vs. Calories) with gender filtering.

Workout type performance analysis via boxplots.

🤖 ML Prediction:

Compare Linear Regression and Random Forest Regressor models.

Real-time prediction based on user inputs (Age, Weight, BPM, etc.).

Feature Importance visualization for the Random Forest model.

📥 Automated Data Loading: Integration with kagglehub to automatically fetch the latest dataset.

🛠️ Tech Stack
Frontend: Streamlit

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn (StandardScaler, LinearRegression, RandomForestRegressor)

Data Sourcing: kagglehub

📦 Installation & Setup
Clone the repository:

Bash
git clone https://github.com/yourusername/fitness-tracker-streamlit.git
cd fitness-tracker-streamlit


2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn kagglehub
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 📂 Dataset
The app is designed to work with the **Fitness Tracker Dataset** (synthetic gym members data). It automatically attempts to download the data using `kagglehub`. Ensure you have an active internet connection on the first run, or place the `gym_members_exercise_tracking_synthetic_data.csv` file in the root directory.

## 🧠 Model Performance
The application trains two models on the fly:
1.  **Linear Regression**: Provides a baseline for linear relationships.
2.  **Random Forest**: Captures non-linear complexities and provides feature importance rankings.

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions for new visualizations or better model tuning.

---
*Built with ❤️ for the Fitness & Data Science community.*
