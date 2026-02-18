# Air Quality Prediction using Machine Learning

## Project Overview
This project focuses on predicting air quality using various machine learning models. The analysis involves data loading, exploratory data analysis, data preprocessing, and training several classification models, including Random Forest and a suite of boosting algorithms (AdaBoost, Gradient Boost, XGBoost, LightGBM, and CatBoost).

## Dataset
The project utilizes a dataset named `updated_pollution_dataset.csv` which contains various environmental parameters such as Temperature, Humidity, PM2.5, PM10, NO2, SO2, CO, Proximity to Industrial Areas, and Population Density, with 'Air Quality' as the target variable.

## Methodology
1.  **Data Loading and Inspection**: The dataset is loaded using pandas, and initial inspections are performed to understand its structure and identify any missing values or data types.
2.  **Data Preprocessing**: Duplicate rows are removed. The target variable 'Air Quality' is separated from the features.
3.  **Model Training and Evaluation**: The data is split into training and testing sets. Several classification models are trained and evaluated based on accuracy, precision, recall, and F1-score. A confusion matrix is also generated for the Random Forest Classifier.

## Models Explored:
-   **Random Forest Classifier**: Achieved high accuracy and robust performance.
-   **Boosting Models**:
    -   **Adaptive Boost (AdaBoost)**: Showed moderate performance, struggling with class balance.
    -   **Gradient Boost (GradientBoostingClassifier)**: Demonstrated strong multi-class performance.
    -   **Extreme Gradient Boost (XGBoost)**: Comparable to Gradient Boost, with reliable performance across classes.
    -   **Light Gradient Boost (LightGBM)**: Achieved the best overall performance, especially for minority classes.
    -   **Categorical Boost (CatBoost)**: Excellent multi-class classifier, handling categorical features well.

## Key Findings
-   The dataset contains 5000 entries with 10 columns, including various environmental factors and an 'Air Quality' target variable.
-   All features are numerical except 'Air Quality', which is categorical.
-   Modern gradient boosting models (LightGBM, CatBoost, XGBoost) consistently outperformed AdaBoost, showing high accuracy and balanced performance across different air quality categories.
-   LightGBM and CatBoost emerged as the top performers with 96% accuracy, demonstrating strong predictive capabilities for both majority and minority classes.

## How to Reproduce
1.  Clone this repository.
2.  Install the required Python packages (see `requirements.txt`).
3.  Ensure `updated_pollution_dataset.csv` is in the same directory as the notebook.
4.  Run the `AirQuality.ipynb` Jupyter notebook.

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
```

