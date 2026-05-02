import kagglehub
path = kagglehub.dataset_download("nadeemajeedch/fitness-tracker-dataset")
import pandas as pd
import numpy as np
import os



path_df=os.path.join(path,"gym_members_exercise_tracking_synthetic_data.csv")
df=pd.read_csv(path_df)



df

df.isnull().sum()

df.shape

df.info()

### Handling Missing Values

Based on the `df.isnull().sum()` output, we have several columns with missing values. We will use the following strategies:

*   **Numerical Columns**: Impute with the `mean` of the column.
*   **Categorical Columns**: Impute with the `mode` (most frequent value) of the column.



# Identify numerical and categorical columns with missing values
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include='object').columns

# Impute numerical columns with the mean
for col in numerical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)

# Impute categorical columns with the mode
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# Verify that there are no more missing values
print("Missing values after imputation:")
display(df.isnull().sum())

df.describe()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df['Calories_Burned'], bins=20, kde=True)
plt.title('Distribution of Calories Burned')
plt.xlabel('Calories Burned')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

### Correlation Heatmap

Let's visualize the correlation matrix of the numerical features using a heatmap. This helps identify highly correlated features, which can be useful for feature selection and understanding underlying relationships.

# Select only numerical columns for correlation calculation
numerical_df = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Plotting the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

### Scatter Plot: Weight vs. Calories Burned

Let's visualize the relationship between 'Weight (kg)' and 'Calories_Burned' using a scatter plot.

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Weight (kg)', y='Calories_Burned', data=df, hue='Gender', palette='viridis', alpha=0.7)
plt.title('Scatter Plot of Weight (kg) vs. Calories Burned')
plt.xlabel('Weight (kg)')
plt.ylabel('Calories Burned')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

### Box Plot: Calories Burned by Gender

Let's visualize the distribution of 'Calories_Burned' for each 'Gender' using a box plot. This will help us compare the central tendency, spread, and presence of outliers in calorie burn between genders.

plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Calories_Burned', data=df, hue='Gender', palette='pastel', legend=False)
plt.title('Distribution of Calories Burned by Gender')
plt.xlabel('Gender')
plt.ylabel('Calories Burned')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

df.duplicated().sum()



### One-Hot Encoding for Categorical Features

To prepare the categorical features ('Gender' and 'Workout_Type') for potential machine learning models, we will apply one-hot encoding. This process converts categorical variables into a numerical format, creating new binary columns for each unique category.

# Reload df to ensure initial state
df = pd.read_csv(path_df)

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include='object').columns

# Impute numerical columns with the mean
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean()) # Using direct assignment to avoid FutureWarning

# Impute categorical columns with the mode
for col in categorical_cols:
    if df[col].isnull().any():
        # Handle cases where all values might be NaN for a categorical column (unlikely here but good practice)
        if not df[col].isnull().all():
            df[col] = df[col].fillna(df[col].mode()[0]) # Using direct assignment to avoid FutureWarning
        else:
            df[col] = df[col].fillna('Unknown') # Fallback if all are NaN

# Clean 'Workout_Type' column: convert to string, strip whitespace, replace literal '\n' and '\t' sequences
df['Workout_Type'] = df['Workout_Type'].astype(str).str.strip()
df['Workout_Type'] = df['Workout_Type'].str.replace('\\n', '', regex=False)
df['Workout_Type'] = df['Workout_Type'].str.replace('\\t', '', regex=False)

# Perform one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'Workout_Type'], drop_first=True)

print("DataFrame after full preprocessing (missing values handled and one-hot encoded):")
display(df.head())
display(df.isnull().sum()) # Verify no more NaNs

x=df.drop(columns=['Calories_Burned'])
y=df['Calories_Burned']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Ensure x_train and x_test are DataFrames for column-based operations
# If they are already numpy arrays (as indicated by the kernel state), convert them back to DataFrame
if not isinstance(x_train, pd.DataFrame):
    x_train_df = pd.DataFrame(x_train, columns=x.columns)
else:
    x_train_df = x_train.copy()

if not isinstance(x_test, pd.DataFrame):
    x_test_df = pd.DataFrame(x_test, columns=x.columns)
else:
    x_test_df = x_test.copy()

# Convert 'Max_BPM' to numeric, coercing errors to NaN
x_train_df['Max_BPM'] = pd.to_numeric(x_train_df['Max_BPM'], errors='coerce')
x_test_df['Max_BPM'] = pd.to_numeric(x_test_df['Max_BPM'], errors='coerce')

# Impute NaN values that resulted from coercion (e.g., with the mean of the training set's column)
mean_max_bpm_train = x_train_df['Max_BPM'].mean()
x_train_df['Max_BPM'] = x_train_df['Max_BPM'].fillna(mean_max_bpm_train)
x_test_df['Max_BPM'] = x_test_df['Max_BPM'].fillna(mean_max_bpm_train) # Use train mean for test set

# Convert back to NumPy arrays for StandardScaler
x_train_processed = x_train_df.values
x_test_processed = x_test_df.values

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train_processed)
x_test_scaled=scaler.transform(x_test_processed)

df.head()

x=df.drop(columns=['Calories_Burned'])
y=df['Calories_Burned']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Ensure x_train and x_test are DataFrames for column-based operations
# If they are already numpy arrays (as indicated by the kernel state), convert them back to DataFrame
if not isinstance(x_train, pd.DataFrame):
    x_train_df = pd.DataFrame(x_train, columns=x.columns)
else:
    x_train_df = x_train.copy()

if not isinstance(x_test, pd.DataFrame):
    x_test_df = pd.DataFrame(x_test, columns=x.columns)
else:
    x_test_df = x_test.copy()

# Convert 'Max_BPM' to numeric, coercing errors to NaN
x_train_df['Max_BPM'] = pd.to_numeric(x_train_df['Max_BPM'], errors='coerce')
x_test_df['Max_BPM'] = pd.to_numeric(x_test_df['Max_BPM'], errors='coerce')

# Impute NaN values that resulted from coercion (e.g., with the mean of the training set's column)
mean_max_bpm_train = x_train_df['Max_BPM'].mean()
x_train_df['Max_BPM'] = x_train_df['Max_BPM'].fillna(mean_max_bpm_train)
x_test_df['Max_BPM'] = x_test_df['Max_BPM'].fillna(mean_max_bpm_train) # Use train mean for test set

# Convert back to NumPy arrays for StandardScaler
x_train_processed = x_train_df.values
x_test_processed = x_test_df.values

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train_processed)
x_test_scaled=scaler.transform(x_test_processed)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Linear Regression model
linear_model = LinearRegression()

# Fit the model to the scaled training data
linear_model.fit(x_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = linear_model.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)

# Fit the model to the scaled training data
rf_model.fit(x_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred_rf = rf_model.predict(x_test_scaled)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Regressor Mean Squared Error: {mse_rf:.2f}")
print(f"Random Forest Regressor R-squared: {r2_rf:.2f}")

### Feature Importance from Random Forest Regressor

To identify the top-performing predictors, we can examine the feature importances calculated by the Random Forest Regressor. This metric indicates the relative contribution of each feature to the model's predictive power.

# Get feature importances from the Random Forest model
feature_importances = rf_model.feature_importances_

# Get feature names from the original DataFrame (before scaling and one-hot encoding)
# Use x.columns from the point x was created before train-test split
feature_names = x.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort features by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Top-performing predictors based on Random Forest Feature Importance:")
display(importance_df)

# Optionally, visualize feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances from Random Forest Regressor')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()