import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from elo import compute_elo

# Load dataset
df = pd.read_csv("data/ufc-master.csv")

# Compute ELO + opponent strength
df = compute_elo(df)

# Recent form
df["WinStreakDif"] = df["RedCurrentWinStreak"] - df["BlueCurrentWinStreak"]
df["LoseStreakDif"] = df["RedCurrentLoseStreak"] - df["BlueCurrentLoseStreak"]

# Remove leakage columns
leak_cols = [
    "RedOdds",
    "BlueOdds",
    "RedExpectedValue",
    "BlueExpectedValue",
    "FinishRound",
    "TotalFightTimeSecs",
    "RedDecOdds",
    "BlueDecOdds",
    "RKOOdds",
    "BKOOdds",
    "RSubOdds",
    "BSubOdds",
]

df = df.drop(columns=leak_cols, errors="ignore")

print("Dataset size:", df.shape)

# Keep only proper labels
df = df[df["Winner"].isin(["Red", "Blue"])].copy()

# Target
y = df["Winner"].apply(lambda x: 1 if x == "Red" else 0)

# Use numeric features only
X = df.select_dtypes(include=["float64", "int64", "int32", "float32"])

# Remove target if present
X = X.drop(columns=["Winner"], errors="ignore")

# Fill missing values
X = X.fillna(0)

print("Features used:", len(X.columns))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = XGBClassifier(
    n_estimators=800,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2,
    min_child_weight=3,
    random_state=42,
    eval_metric="logloss"
)

# Train
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Test Accuracy:", acc)

scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Accuracy:", scores.mean())

# Top features
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

print("\nTop 15 Important Features:")
print(top_features)

# Save
joblib.dump(model, "models/ufc_model.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")
joblib.dump(top_features.index.tolist(), "models/top_features.pkl")

print("\nModel saved to models/ufc_model.pkl")
print("Feature names saved to models/feature_names.pkl")
print("Top features saved to models/top_features.pkl")