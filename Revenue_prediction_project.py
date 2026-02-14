#create machine learning model to predict revenue of the restaurant based on the features present in the dataset (built it by Linear Regression)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv(r"C:\Users\hardi\Downloads\revenue_prediction (1).csv")

df

# Analysing the data

print(df.head())
print(df.tail())

print(df.shape)

print(df.info())
print(df.describe())

print(df.isna().sum())

# total number of rows with null values

n= df.isna().any(axis=1).sum()
print(n)

print(df.duplicated().sum())

for i in df.columns:
    if df[i].dtypes == "int" or df[i].dtypes == "float":
        sns.boxplot(df[i])
        plt.xlabel(i)
        plt.show()

# cleaning the data

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print(df.dtypes)

# label encoding
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
Le = LabelEncoder()
for i in df.columns:
    if df[i].dtypes == "object":
        df[i] = Le.fit_transform(df[i])
x = df.drop(['Revenue', 'Id', 'Name'], axis=1)
y = df['Revenue']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=0)

# feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Residuals plot for Linear Regression
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Revenue')
plt.ylabel('Residuals')
plt.title('Residuals Plot for Linear Regression')
plt.show()

# Visualizations
# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(x.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot for features
sns.pairplot(df.drop(['Id', 'Name'], axis=1))
plt.show()

# Apply VIF to detect multicollinearity
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

vif_df = calculate_vif(x)
print("VIF DataFrame:")
print(vif_df)

# Remove features with VIF > 5
high_vif = vif_df[vif_df['VIF'] > 5]['feature'].tolist()
x = x.drop(high_vif, axis=1)
print("Features after VIF removal:", x.columns.tolist())

# Re-split after VIF
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Re-scale
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Re-fit Linear Regression after VIF
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Linear Regression R2 after VIF:", r2_score(y_test, y_pred))

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
model_pca = LinearRegression()
model_pca.fit(x_train_pca, y_train)
y_pred_pca = model_pca.predict(x_test_pca)
print("PCA Linear Regression R2:", r2_score(y_test, y_pred_pca))

# Apply LDA
# Bin the target into 3 classes for LDA
y_train_binned = pd.cut(y_train, bins=3, labels=[0, 1, 2])
y_test_binned = pd.cut(y_test, bins=3, labels=[0, 1, 2])
lda = LinearDiscriminantAnalysis(n_components=2)  # Max components for 3 classes is 2
x_train_lda = lda.fit_transform(x_train, y_train_binned)
x_test_lda = lda.transform(x_test)
model_lda = LinearRegression()
model_lda.fit(x_train_lda, y_train)
y_pred_lda = model_lda.predict(x_test_lda)
print("LDA Linear Regression R2:", r2_score(y_test, y_pred_lda))


# Random Forest with RandomizedSearchCV
rf = RandomForestRegressor(random_state=0)
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=50, cv=5, scoring='r2', random_state=0, n_jobs=-1)
random_search_rf.fit(x_train, y_train)
best_rf = random_search_rf.best_estimator_
y_pred_rf = best_rf.predict(x_test)
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

# Compare models
models_r2 = {
    'Linear (after VIF)': r2_score(y_test, y_pred),
    'PCA Linear': r2_score(y_test, y_pred_pca),
    'LDA Linear': r2_score(y_test, y_pred_lda), 
    'Random Forest': r2_score(y_test, y_pred_rf)
}
best_model = max(models_r2, key=models_r2.get)
print(f"Best model: {best_model} with R2 Score: {models_r2[best_model]}")

