# Revenue Prediction Project

A Machine Learning project focused on predicting restaurant revenue based on specific features using Linear Regression and Ensemble methods.

## 🚀 Project Overview
This project involves building a predictive model to estimate revenue. It follows a rigorous data science pipeline, including exploratory data analysis (EDA), data cleaning, feature engineering, and model optimization.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn, Statsmodels

## 📊 Key Features & Methodology
To ensure high model accuracy and reliability, the following techniques were implemented:

* **Data Cleaning:** Handled missing values and removed duplicates to ensure data quality.
* **Exploratory Data Analysis (EDA):** Used boxplots for outlier detection and heatmaps for correlation analysis.
* **Feature Engineering:**
    * **Label Encoding:** Converted categorical variables into numerical format.
    * **Standard Scaling:** Normalized features for better model convergence.
* **Advanced Optimization:**
    * **Multicollinearity Check:** Applied **Variance Inflation Factor (VIF)** to remove redundant features.
    * **Dimensionality Reduction:** Utilized **PCA (Principal Component Analysis)** to retain 95% variance.
    * **Hyperparameter Tuning:** Used **RandomizedSearchCV** to optimize the Random Forest Regressor.

## 📈 Model Performance
We compared multiple models to find the best fit:
* **Linear Regression (Post-VIF)**
* **PCA-based Linear Regression**
* **LDA-based Linear Regression**
* **Random Forest Regressor** (Best Performing)

## 📁 Repository Structure
* `Revenue_prediction_project.py`: Main Python script containing the model pipeline.
* `revenue_prediction.csv`: The dataset used for training and testing.
* `README.md`: Project documentation.
