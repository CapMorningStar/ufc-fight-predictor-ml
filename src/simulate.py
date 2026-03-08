import pandas as pd
import joblib
import numpy as np
import os

# 1. Load the model and feature names
# Using the 'models/' path to match your new train.py
model_path = os.path.join("models", "ufc_model.pkl")
features_path = os.path.join("models", "feature_names.pkl")
tonight_path = os.path.join("data", "max_charles_tonight.csv")

model = joblib.load(model_path)
feature_names = joblib.load(features_path)
df_tonight = pd.read_csv(tonight_path)

# 2. Base Data with ALL required features
base_data = {
    "RedAge": df_tonight.loc[0, "RedAge"],
    "BlueAge": df_tonight.loc[0, "BlueAge"],
    "RedHeightCms": df_tonight.loc[0, "RedHeightCms"],
    "BlueHeightCms": df_tonight.loc[0, "BlueHeightCms"],
    "RedReachCms": df_tonight.loc[0, "RedReachCms"],
    "BlueReachCms": df_tonight.loc[0, "BlueReachCms"],
    "RedAvgSigStrLanded": df_tonight.loc[0, "RedAvgSigStrLanded"],
    "BlueAvgSigStrLanded": df_tonight.loc[0, "BlueAvgSigStrLanded"],
    "RedAvgTDLanded": df_tonight.loc[0, "RedAvgTDLanded"],
    "BlueAvgTDLanded": df_tonight.loc[0, "BlueAvgTDLanded"],
    "RedAvgSubAtt": 0.3,   # Holloway career avg
    "BlueAvgSubAtt": 2.8,  # Oliveira career avg
    "AgeDif": df_tonight.loc[0, "AgeDif"],
    "HeightDif": df_tonight.loc[0, "HeightDif"],
    "ReachDif": df_tonight.loc[0, "ReachDif"],
    "SigStrDif": df_tonight.loc[0, "SigStrDif"],
    "AvgTDDif": df_tonight.loc[0, "AvgTDDif"],
    "RedELO": 1750,
    "BlueELO": 1715,
    "WinStreakDif": 1,
    "LoseStreakDif": -1,
}

n_sims = 10000
red_wins = 0
blue_wins = 0

print(f"Starting {n_sims} simulations for Holloway vs Oliveira...")

for _ in range(n_sims):
    sim_data = base_data.copy()

    # 3. Add randomness to the 'Performance' stats
    # We add noise to the base stats, then RE-CALCULATE the Difs
    noise_scale = 0.1 
    sim_data["RedAvgSigStrLanded"] += np.random.normal(0, noise_scale)
    sim_data["BlueAvgSigStrLanded"] += np.random.normal(0, noise_scale)
    sim_data["RedAvgTDLanded"] += np.random.normal(0, noise_scale)
    sim_data["BlueAvgTDLanded"] += np.random.normal(0, noise_scale)
    
    # 4. Re-calculate the new features based on the noisy stats
    sim_data["RedGrapplingScore"] = sim_data["RedAvgTDLanded"] + (sim_data["RedAvgSubAtt"] * 2)
    sim_data["BlueGrapplingScore"] = sim_data["BlueAvgTDLanded"] + (sim_data["BlueAvgSubAtt"] * 2)
    sim_data["GrapplingPressureDif"] = sim_data["RedGrapplingScore"] - sim_data["BlueGrapplingScore"]
    
    # Update Difs to stay consistent with noise
    sim_data["SigStrDif"] = sim_data["RedAvgSigStrLanded"] - sim_data["BlueAvgSigStrLanded"]
    sim_data["AvgTDDif"] = sim_data["RedAvgTDLanded"] - sim_data["BlueAvgTDLanded"]

    # 5. Format and Predict
    X_sim = pd.DataFrame([sim_data])
    
    # Ensure all features match training
    for col in feature_names:
        if col not in X_sim.columns:
            X_sim[col] = 0
    X_sim = X_sim[feature_names]

    probs = model.predict_proba(X_sim)[0]
    
    # Choose winner based on probability
    winner = np.random.choice([1, 0], p=[probs[1], probs[0]])

    if winner == 1:
        red_wins += 1
    else:
        blue_wins += 1

print("-" * 30)
print(f"RESULTS AFTER {n_sims} FIGHTS:")
print(f"Max Holloway (Red) Win Rate: {red_wins / n_sims:.2%}")
print(f"Charles Oliveira (Blue) Win Rate: {blue_wins / n_sims:.2%}")