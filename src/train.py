import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Import your local ELO module
from elo import compute_elo

# 1. Load dataset
data_path = os.path.join("data", "ufc-master.csv")
df = pd.read_csv(data_path)

# 2. Compute ELO + opponent strength
df = compute_elo(df)

# 3. Feature Engineering
# Recent form
df["WinStreakDif"] = df["RedCurrentWinStreak"] - df["BlueCurrentWinStreak"]
df["LoseStreakDif"] = df["RedCurrentLoseStreak"] - df["BlueCurrentLoseStreak"]

# --- NEW: Grappling Pressure Feature ---
# This combines Takedowns and Submissions into one 'Grappling' score
# We multiply Submissions by 2 to give them higher weight in the 'danger' factor
df["RedGrapplingScore"] = df["RedAvgTDLanded"] + (df["RedAvgSubAtt"] * 2)
df["BlueGrapplingScore"] = df["BlueAvgTDLanded"] + (df["BlueAvgSubAtt"] * 2)
df["GrapplingPressureDif"] = df["RedGrapplingScore"] - df["BlueGrapplingScore"]

# 4. Remove leakage columns
leak_cols = [
    "RedOdds", "BlueOdds", "RedExpectedValue", "BlueExpectedValue",
    "FinishRound", "TotalFightTimeSecs", "RedDecOdds", "BlueDecOdds",
    "RKOOdds", "BKOOdds", "RSubOdds", "BSubOdds",
]
df = df.drop(columns=leak_cols, errors="ignore")

# 5. Prepare Data for Training
# Keep only proper labels
df = df[df["Winner"].isin(["Red", "Blue"])].copy()

# Target
y = df["Winner"].apply(lambda x: 1 if x == "Red" else 0)

# Use numeric features only
X = df.select_dtypes(include=["float64", "int64", "int32", "float32"])
X = X.drop(columns=["Winner"], errors="ignore")
X = X.fillna(0)

# Save feature names so predict.py knows the exact order
os.makedirs("models", exist_ok=True)
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print(f"Dataset size: {df.shape}")
print(f"Features used: {len(X.columns)}")

# 6. Split and Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=1000, # Increased for better learning
    max_depth=4,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2,
    min_child_weight=3,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# 7. Evaluate and Save
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {acc:.4f}")

joblib.dump(model, "models/ufc_model.pkl")

# Save Top Features for analysis
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
joblib.dump(importances.head(20).index.tolist(), "models/top_features.pkl")

print("✅ Model updated and saved in 'models/' folder.")