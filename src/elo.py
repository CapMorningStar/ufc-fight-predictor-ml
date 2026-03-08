import pandas as pd

def compute_elo(df, k=32):
    elo = {}
    opponent_avg_elo_red = []
    opponent_avg_elo_blue = []
    red_elos = []
    blue_elos = []

    for _, row in df.iterrows():
        red = row["RedFighter"]
        blue = row["BlueFighter"]

        red_rating = elo.get(red, 1500)
        blue_rating = elo.get(blue, 1500)

        red_elos.append(red_rating)
        blue_elos.append(blue_rating)

        # opponent strength before the fight
        opponent_avg_elo_red.append(blue_rating)
        opponent_avg_elo_blue.append(red_rating)

        expected_red = 1 / (1 + 10 ** ((blue_rating - red_rating) / 400))
        expected_blue = 1 - expected_red

        if row["Winner"] == "Red":
            score_red = 1
            score_blue = 0
        elif row["Winner"] == "Blue":
            score_red = 0
            score_blue = 1
        else:
            score_red = 0.5
            score_blue = 0.5

        elo[red] = red_rating + k * (score_red - expected_red)
        elo[blue] = blue_rating + k * (score_blue - expected_blue)

    df["RedELO"] = red_elos
    df["BlueELO"] = blue_elos
    df["ELODif"] = df["RedELO"] - df["BlueELO"]

    df["RedOpponentStrength"] = opponent_avg_elo_red
    df["BlueOpponentStrength"] = opponent_avg_elo_blue
    df["OpponentStrengthDif"] = (
        df["RedOpponentStrength"] - df["BlueOpponentStrength"]
    )

    return df