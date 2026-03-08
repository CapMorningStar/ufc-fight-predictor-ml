import pandas as pd
import os

# 1. Point to the 'data' folder
master_path = os.path.join("data", "ufc-master.csv")
tonight_path = os.path.join("data", "max_charles_tonight.csv")

# 2. Safety Check: If the files aren't there, tell the user why
if not os.path.exists(master_path):
    print(f"❌ ERROR: Could not find {master_path}")
    print("Make sure you are running the script from the main 'ufc-fight-predictor' folder.")
    exit()

# 3. Load the data
master_df = pd.read_csv(master_path)
tonight_df = pd.read_csv(tonight_path)

# 4. Create the new row for the Holloway vs Oliveira result
new_row = pd.Series(index=master_df.columns, dtype='object')

new_row['RedFighter'] = "Max Holloway"
new_row['BlueFighter'] = "Charles Oliveira"
new_row['Winner'] = "Blue"  # The actual result
new_row['Date'] = "2026-03-07"
new_row['TitleBout'] = True
new_row['WeightClass'] = "Lightweight"

# 5. Map the pre-fight stats
for col in tonight_df.columns:
    if col in master_df.columns:
        new_row[col] = tonight_df.iloc[0][col]

# 6. Append and Save
master_df = pd.concat([master_df, pd.DataFrame([new_row])], ignore_index=True)
master_df.to_csv(master_path, index=False)

print(f"✅ SUCCESS: Added UFC 326 result to {master_path}!")