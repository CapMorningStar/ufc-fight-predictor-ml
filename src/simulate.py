import pandas as pd
import joblib
import numpy as np

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

n_sims = 10000
red_wins = 0
blue_wins = 0

for _ in range(n_sims):
    sim_data = base_data.copy()

    # add small randomness
    for key in [
        "RedAvgSigStrLanded", "BlueAvgSigStrLanded",
        "RedAvgTDLanded", "BlueAvgTDLanded",
        "RedELO", "BlueELO", "ELODif",
        "WinStreakDif", "OpponentStrengthDif"
    ]:
        sim_data[key] += np.random.normal(0, 0.5)

    X_sim = pd.DataFrame([sim_data])

    for col in feature_names:
        if col not in X_sim.columns:
            X_sim[col] = 0

    X_sim = X_sim[feature_names]
    probs = model.predict_proba(X_sim)[0]

    red_prob = probs[1]
    blue_prob = probs[0]

    winner = np.random.choice([1, 0], p=[red_prob, blue_prob])

    if winner == 1:
        red_wins += 1
    else:
        blue_wins += 1

print("Simulations:", n_sims)
print("Red win rate:", red_wins / n_sims)
print("Blue win rate:", blue_wins / n_sims)

# rough finish-type estimate based on style
red_ko_prob = min(max((base_data["SigStrDif"] + 5) / 20, 0.10), 0.55)
blue_sub_prob = min(max(((-base_data["AvgTDDif"]) + 3) / 15, 0.10), 0.45)
decision_prob = max(0.20, 1 - red_ko_prob - blue_sub_prob)

print("\nRough style-based finish estimates:")
print("Red KO/TKO probability:", round(red_ko_prob, 3))
print("Blue submission probability:", round(blue_sub_prob, 3))
print("Decision probability:", round(decision_prob, 3))