import pandas as pd
import joblib
import shap

model = joblib.load("models/ufc_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

df = pd.read_csv("data/max_charles_tonight.csv")

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
    "AgeDif": df.loc[0, "AgeDif"],
    "HeightDif": df.loc[0, "HeightDif"],
    "ReachDif": df.loc[0, "ReachDif"],
    "SigStrDif": df.loc[0, "SigStrDif"],
    "AvgTDDif": df.loc[0, "AvgTDDif"],
    "RedELO": 1750,
    "BlueELO": 1715,
    "ELODif": 35,
    "RedOpponentStrength": 1720,
    "BlueOpponentStrength": 1705,
    "OpponentStrengthDif": 15,
    "WinStreakDif": 1,
    "LoseStreakDif": -1,
}

X_new = pd.DataFrame([base_data])

for col in feature_names:
    if col not in X_new.columns:
        X_new[col] = 0

X_new = X_new[feature_names]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_new)

shap_series = pd.Series(shap_values[0], index=X_new.columns).sort_values(
    key=abs, ascending=False
)

print("Top 15 features influencing this prediction:\n")
print(shap_series.head(15))