# ============================
# IMPORT LIBRARIES
# ============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report


# ============================
# LOAD DATASET
# ============================

df = pd.read_excel("financialbehaviour.xlsx")

# Drop personal columns
df = df.drop(columns=["Timestamp", "Email Address"])

print(df.head())
print(df.info())

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("\nColumns after strip:\n", df.columns)


print("\n================ REGRESSION: Miscellaneous expenses prediction =================")

# Column names based on the actual dataset
col_shopping = "6. Average monthly spending on shopping (₹)"
col_food     = "7. Average monthly spending on food/eating out (₹)"
col_travel   = "8. Average monthly spending on Traveling (₹)"
col_misc     = "9.Miscellaneous expenses"
col_income   = "3. Monthly Allowance or Income"



# 🎯 Target column
target_total = col_misc

# ✅ Feature(s): using only Monthly Income (no leakage from target)
feature_cols_total = [
    col_income,col_food,col_shopping,col_travel
]

# ✅ Remove rows with missing values for these columns
df_total = df.dropna(subset=feature_cols_total + [target_total])

X_tot = df_total[feature_cols_total].copy()
y_tot = df_total[target_total].copy()

# ✅ Ensure numeric types
for col in feature_cols_total:
    X_tot[col] = pd.to_numeric(X_tot[col], errors="coerce")

y_tot = pd.to_numeric(y_tot, errors="coerce")

# ✅ Remove invalid rows (NaNs after conversion)
valid_idx_tot = X_tot.notna().all(axis=1) & y_tot.notna()
X_tot = X_tot[valid_idx_tot]
y_tot = y_tot[valid_idx_tot]

print("\nShape for regression X, y:", X_tot.shape, y_tot.shape)


# ============================
# SCALE FEATURES
# ============================

scaler_tot = StandardScaler()
X_tot_scaled = scaler_tot.fit_transform(X_tot)

# ============================
# TRAIN–TEST SPLIT
# ============================

X_tot_train, X_tot_test, y_tot_train, y_tot_test = train_test_split(
    X_tot_scaled, y_tot, test_size=0.3, random_state=100
)

# ---------------------------
# ✅ 1️⃣ MULTIPLE LINEAR REGRESSION
# ---------------------------

lr_tot = LinearRegression()
lr_tot.fit(X_tot_train, y_tot_train)
pred_tot_lr = lr_tot.predict(X_tot_test)

print("\n✅ Multiple Linear Regression (Monthly Total Expenses)")
print("MSE :", mean_squared_error(y_tot_test, pred_tot_lr))
print("MAE :", mean_absolute_error(y_tot_test, pred_tot_lr))
print("R2  :", r2_score(y_tot_test, pred_tot_lr))

# ---------------------------
# ✅ 2️⃣ POLYNOMIAL REGRESSION (DEGREE 2)
# ---------------------------

poly_tot = PolynomialFeatures(degree=2)
X_tot_train_poly = poly_tot.fit_transform(X_tot_train)
X_tot_test_poly = poly_tot.transform(X_tot_test)

poly_model_tot = LinearRegression()
poly_model_tot.fit(X_tot_train_poly, y_tot_train)
pred_tot_poly = poly_model_tot.predict(X_tot_test_poly)

print("\n✅ Polynomial Regression (Degree 2) - Monthly Total Expenses")
print("MSE :", mean_squared_error(y_tot_test, pred_tot_poly))
print("MAE :", mean_absolute_error(y_tot_test, pred_tot_poly))
print("R2  :", r2_score(y_tot_test, pred_tot_poly))

# ---------------------------
# ✅ 3️⃣ RANDOM FOREST REGRESSION
# ---------------------------

rf_tot = RandomForestRegressor(
    n_estimators=800,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

rf_tot.fit(X_tot_train, y_tot_train)
pred_tot_rf = rf_tot.predict(X_tot_test)

print("\n✅ ✅ ✅ Random Forest Regression - Monthly Total Expenses")
print("MSE :", mean_squared_error(y_tot_test, pred_tot_rf))
print("MAE :", mean_absolute_error(y_tot_test, pred_tot_rf))
print("R2  :", r2_score(y_tot_test, pred_tot_rf))


# ============================================================
#2. CLASSIFICATION: HIGH SAVER PREDICTION
# ============================================================

print("\n================ CLASSIFICATION: HIGH SAVER =================")

# 🔹 Correct column name in your dataset:
# '13. Approximate percentage of your income you save monthly (%)'
target2 = "13. Approximate percentage of your income you save monthly (%)"

# Create binary label: 1 if save >=10%, else 0
df["High_Saver"] = df[target2].apply(lambda x: 1 if x >= 10 else 0)

X2 = df.drop(columns=[target2, "High_Saver"])
y2 = df["High_Saver"]

# One-hot encode categorical features
X2 = pd.get_dummies(X2, drop_first=True)

# Scale all features
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_scaled, y2, test_size=0.2, random_state=42
)

# ---------------------------
# ✅ LOGISTIC REGRESSION
# ---------------------------
log = LogisticRegression(max_iter=1000)
log.fit(X2_train, y2_train)
pred_log = log.predict(X2_test)

print("\n--- Logistic Regression (High Saver) ---")
print("Accuracy :", accuracy_score(y2_test, pred_log))
print("Precision:", precision_score(y2_test, pred_log))
print("Confusion Matrix:\n", confusion_matrix(y2_test, pred_log))
print(classification_report(y2_test, pred_log))

# ---------------------------
# KNN
# ---------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X2_train, y2_train)
pred_knn = knn.predict(X2_test)

print("\n--- KNN (High Saver) ---")
print("Accuracy :", accuracy_score(y2_test, pred_knn))
print("Precision:", precision_score(y2_test, pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y2_test, pred_knn))
print(classification_report(y2_test, pred_knn))

# ---------------------------
# DECISION TREE
# ---------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X2_train, y2_train)
pred_dt = dt.predict(X2_test)

print("\n--- Decision Tree (High Saver) ---")
print("Accuracy :", accuracy_score(y2_test, pred_dt))
print("Precision:", precision_score(y2_test, pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y2_test, pred_dt))
print(classification_report(y2_test, pred_dt))

# ---------------------------
# SVM
# ---------------------------
svm = SVC()
svm.fit(X2_train, y2_train)
pred_svm = svm.predict(X2_test)

print("\n--- SVM (High Saver) ---")
print("Accuracy :", accuracy_score(y2_test, pred_svm))
print("Precision:", precision_score(y2_test, pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y2_test, pred_svm))
print(classification_report(y2_test, pred_svm))


# ============================================================
# ✅ ✅ ✅ 3. CLASSIFICATION: SPENDING CATEGORY
# ============================================================

print("\n================ CLASSIFICATION: SPENDING CATEGORY =================")

# 🔹 Correct column name in your dataset:
# '10. Category spent on most'
target3 = "10. Category spent on most"

X3 = df.drop(columns=[target3, "High_Saver"])
y3 = df[target3]

# One-hot encode categorical features
X3 = pd.get_dummies(X3, drop_first=True)

# Scale
scaler3 = StandardScaler()
X3_scaled = scaler3.fit_transform(X3)

X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3_scaled, y3, test_size=0.2, random_state=42
)

# ---------------------------
# ✅ LOGISTIC REGRESSION
# ---------------------------
log3 = LogisticRegression(max_iter=1000)
log3.fit(X3_train, y3_train)
pred_log3 = log3.predict(X3_test)

print("\n--- Logistic Regression (Spending Category) ---")
print("Accuracy :", accuracy_score(y3_test, pred_log3))
print("Confusion Matrix:\n", confusion_matrix(y3_test, pred_log3))
print(classification_report(y3_test, pred_log3))

# ---------------------------
# KNN
# ---------------------------
knn3 = KNeighborsClassifier(n_neighbors=5)
knn3.fit(X3_train, y3_train)
pred_knn3 = knn3.predict(X3_test)

print("\n--- KNN (Spending Category) ---")
print("Accuracy :", accuracy_score(y3_test, pred_knn3))
print("Confusion Matrix:\n", confusion_matrix(y3_test, pred_knn3))
print(classification_report(y3_test, pred_knn3))

# ---------------------------
# DECISION TREE
# ---------------------------
dt3 = DecisionTreeClassifier(random_state=42)
dt3.fit(X3_train, y3_train)
pred_dt3 = dt3.predict(X3_test)

print("\n--- Decision Tree (Spending Category) ---")
print("Accuracy :", accuracy_score(y3_test, pred_dt3))
print("Confusion Matrix:\n", confusion_matrix(y3_test, pred_dt3))
print(classification_report(y3_test, pred_dt3))

# ---------------------------
# SVM
# ---------------------------
svm3 = SVC()
svm3.fit(X3_train, y3_train)
pred_svm3 = svm3.predict(X3_test)

print("\n--- SVM (Spending Category) ---")
print("Accuracy :", accuracy_score(y3_test, pred_svm3))
print("Confusion Matrix:\n", confusion_matrix(y3_test, pred_svm3))
print(classification_report(y3_test, pred_svm3))



# ============================
# VISUALIZATION SECTION 
# ============================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

sns.set(style="whitegrid")

# -----------------------------------------
# (3) CORRELATION HEATMAP
# -----------------------------------------
plt.figure(figsize=(10,7))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------------------
# (4) PAIRPLOT (Selected Numeric Columns)
# -----------------------------------------
pair_cols = [
    "3. Monthly Allowance or Income",
    "6. Average monthly spending on shopping (₹)",
    "7. Average monthly spending on food/eating out (₹)",
    "8. Average monthly spending on Traveling (₹)",
    "9.Miscellaneous expenses",
    "13. Approximate percentage of your income you save monthly (%)"
]

pair_cols = [c for c in pair_cols if c in df.columns]

sns.pairplot(df[pair_cols].dropna(), diag_kind="kde")
plt.suptitle("Pairplot – Selected Numeric Features", y=1.02)
plt.show()

# -----------------------------------------
# (5) SPENDING CATEGORY DISTRIBUTION
# -----------------------------------------
plt.figure(figsize=(8,5))
df["10. Category spent on most"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Spending Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# -----------------------------------------
# (7) REGRESSION: ACTUAL vs PREDICTED for ALL MODELS
# -----------------------------------------

models_pred = {
    "Linear Regression": pred_tot_lr,
    "Polynomial Regression (Deg 2)": pred_tot_poly,
    "Random Forest Regression": pred_tot_rf
}

plt.figure(figsize=(15,4))

for i, (name, pred) in enumerate(models_pred.items(), 1):
    plt.subplot(1, 3, i)
    plt.scatter(y_tot_test, pred, alpha=0.6)
    minv = min(y_tot_test.min(), pred.min())
    maxv = max(y_tot_test.max(), pred.max())
    plt.plot([minv, maxv], [minv, maxv], 'r--')
    plt.title(name)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

plt.suptitle("Actual vs Predicted (Regression Models)")
plt.show()

# -----------------------------------------
# (13) HIGH SAVER DISTRIBUTION BAR PLOT
# -----------------------------------------
plt.figure(figsize=(6,4))
df["High_Saver"].value_counts().sort_index().plot(kind="bar", color=["orange","green"])
plt.xticks([0,1], ["Low Saver (<10%)", "High Saver (>=10%)"], rotation=0)
plt.title("High Saver Distribution")
plt.ylabel("Count")
plt.show()

# -----------------------------------------
# (11) CONFUSION MATRICES FOR ALL BINARY CLASSIFICATION MODELS
# -----------------------------------------
binary_models = {
    "Logistic Regression": log,
    "KNN": knn,
    "Decision Tree": dt,
    "SVM": svm
}

plt.figure(figsize=(12,10))

for i, (name, model) in enumerate(binary_models.items(), 1):
    preds = model.predict(X2_test)
    cm = confusion_matrix(y2_test, preds)
    plt.subplot(2, 2, i)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# =============================================
# CONFUSION MATRICES — SPENDING CATEGORY (MULTI-CLASS)
# =============================================



multi_models = {
    "Logistic Regression (Multi-Class)": log3,
    "KNN (Multi-Class)": knn3,
    "Decision Tree (Multi-Class)": dt3,
    "SVM (Multi-Class)": svm3
}

unique_classes = np.unique(y3_test)

plt.figure(figsize=(12,10))

for i, (name, model) in enumerate(multi_models.items(), 1):
    preds = model.predict(X3_test)
    cm = confusion_matrix(y3_test, preds, labels=unique_classes)

    plt.subplot(2, 2, i)
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=unique_classes,
                yticklabels=unique_classes)

    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.show()
