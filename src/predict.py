import pandas as pd
import joblib
import os

# 1. Load the new model and feature list
model = joblib.load("models/ufc_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# 2. Load the pre-fight stats
df = pd.read_csv("data/max_charles_tonight.csv")

# 3. Build the base data dictionary
base_data = {
    "RedAge": df.loc[0, "RedAge"],
    "BlueAge": df.loc[0, "BlueAge"],
    "RedHeightCms": df.loc[0, "RedHeightCms"],
    "BlueHeightCms": df.loc[0, "BlueHeightCms"],
    "RedReachCms": df.loc[0, "RedReachCms"],
    "BlueReachCms": df.loc[0, "BlueReachCms"],
    "RedAvgSigStrLanded": df.loc[0, "RedAvgSigStrLanded"],
    "BlueAvgSigStrLanded": df.loc[0, "BlueAvgSigStrLanded"],
    "RedAvgTDLanded": df.loc[0, "RedAvgTDLanded"],
    "BlueAvgTDLanded": df.loc[0, "BlueAvgTDLanded"],
    "RedAvgSubAtt": 0.3, # Max's career average
    "BlueAvgSubAtt": 2.8, # Charles's career average
    "AgeDif": df.loc[0, "AgeDif"],
    "HeightDif": df.loc[0, "HeightDif"],
    "ReachDif": df.loc[0, "ReachDif"],
    "SigStrDif": df.loc[0, "SigStrDif"],
    "AvgTDDif": df.loc[0, "AvgTDDif"],
    "RedELO": 1750, 
    "BlueELO": 1715,
    "WinStreakDif": 1,
    "LoseStreakDif": -1,
}

# --- NEW: Match the Training Feature Logic ---
base_data["RedGrapplingScore"] = base_data["RedAvgTDLanded"] + (base_data["RedAvgSubAtt"] * 2)
base_data["BlueGrapplingScore"] = base_data["BlueAvgTDLanded"] + (base_data["BlueAvgSubAtt"] * 2)
base_data["GrapplingPressureDif"] = base_data["RedGrapplingScore"] - base_data["BlueGrapplingScore"]

# 4. Format for prediction
X_pred = pd.DataFrame([base_data])

# Ensure columns match the training order exactly
for col in feature_names:
    if col not in X_pred.columns:
        X_pred[col] = 0
X_pred = X_pred[feature_names]

# 5. Get Probability
probs = model.predict_proba(X_pred)[0]
print(f"\nUpdated Prediction for Holloway vs Oliveira:")
print(f"Red (Max) Win Prob: {probs[1]:.2f}")
print(f"Blue (Charles) Win Prob: {probs[0]:.2f}")