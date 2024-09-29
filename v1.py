import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# 1. Reading data from CSV
def read_csv(file_path):
    return pd.read_csv(file_path)


# 2. Getting information and statistics about the dataset
def dataset_info_statistics(data):
    print("Dataset Information:")
    print(data.info())
    print("\nBasic Statistics for Numerical Columns:")
    print(data.describe())
    print("\n")


# 3. Check for null values in the dataset
def check_null(data):
    return data.isnull().sum()


# 4. Check for duplicated rows in the dataset
def check_duplicates(data):
    return data.duplicated().any()


# 5. Plotting graphs for numerical and categorical columns
def plot_graph(data):
    numerical_columns = data.select_dtypes(include=np.number).columns
    for column in numerical_columns:
        plt.figure(figsize=(5, 3))
        sns.distplot(data[column], kde=True)
        plt.title(f"Histogram for {column}")
        plt.show()

    categorical_columns = data.select_dtypes(include='object').columns
    for column in categorical_columns:
        plt.figure(figsize=(5, 3))
        sns.countplot(data[column])
        plt.title(f'Countplot for {column}')
        plt.xticks(rotation=45)
        plt.show()


# 6. Separate features and target
def separate_features_target(data, target_column):
    X = data.drop(columns=[target_column], axis=1)
    y = data[target_column]
    return X, y


# 7. Perform train-test split
def perform_train_test_split(X, y, test_size=0.20, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Read the data
calories = read_csv('calories.csv')
exercise = read_csv('exercise.csv')
data = pd.merge(calories, exercise, on='User_ID')

# Prepare the data
X, y = separate_features_target(data, 'Calories')
X = X.drop(columns=['User_ID'])

X_train, X_test, y_train, y_test = perform_train_test_split(X, y)

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('ordinal', OrdinalEncoder(), ['Gender']),
    ('num', StandardScaler(), ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']),
], remainder='passthrough')


# Model scorer function
def model_scorer(model_name, model):
    output = [model_name]
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = perform_train_test_split(X, y)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    output.append(r2_score(y_test, y_pred))
    output.append(mean_absolute_error(y_test, y_pred))

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')
    output.append(cv_results.mean())

    return output


# Models to test
model_dict = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
}

# Evaluate models
model_output = []
for model_name, model in model_dict.items():
    model_output.append(model_scorer(model_name, model))

print(model_output)

# Final model
pipeline1 = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor())
])

pipeline1.fit(X, y)

# Sample prediction input
gender = input("Enter your gender: ")
age = int(input("Enter your age: "))
Height = float(input("Enter your height: "))
Weight = float(input("Enter your weight: "))
Duration = float(input("Enter your duration: "))
Heart_Rate = float(input("Enter your heart rate: "))
Body_Temp = float(input("Enter your body temp: "))

sample = pd.DataFrame({
    'Gender': gender,
    'Age': age,
    'Height': Height,
    'Weight': Weight,
    'Duration': Duration,
    'Heart_Rate': Heart_Rate,
    'Body_Temp': Body_Temp
}, index=[0])

# Predicting using the trained pipeline
predicted_calories = pipeline1.predict(sample)
print(f"Predicted calories burned: {predicted_calories[0]}")
