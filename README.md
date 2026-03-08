# UFC Fight Predictor using Machine Learning

This project is a **machine learning system that predicts UFC fight
outcomes** using historical fight data, fighter statistics, ELO ratings,
and Monte Carlo simulations.

The goal of this project is to demonstrate how **sports analytics and
machine learning** can be used to estimate fight probabilities and
analyze the factors that influence fight outcomes.

------------------------------------------------------------------------

# Project Overview

This system analyzes thousands of past UFC fights and trains a model
that learns patterns such as:

-   Fighter striking volume
-   Reach advantage
-   Takedown ability
-   Win streak momentum
-   Fighter skill rating (ELO)
-   Opponent strength

Using these signals, the model predicts the probability of one fighter
beating another.

------------------------------------------------------------------------

# Key Components

## Fighter Statistics

The model uses historical statistics such as:

-   Age
-   Height
-   Reach
-   Strikes landed per minute
-   Takedowns per fight
-   Win/loss record
-   Win streaks

These features help describe how fighters perform in the octagon.

------------------------------------------------------------------------

## ELO Rating System

Each fighter has an **ELO rating**, which represents their overall skill
level based on past fights.

How it works:

-   Fighters start around **1500 ELO**
-   Winning increases rating
-   Losing decreases rating
-   Beating stronger opponents increases rating more

Example:

Volkanovski: 1840\
Holloway: 1780

This allows the model to understand **fighter strength over time**.

------------------------------------------------------------------------

## Opponent Strength

The system also tracks the strength of opponents each fighter has faced.

Example:

Beating a champion is more impressive than beating a newcomer.

This helps measure **strength of schedule**.

------------------------------------------------------------------------

## Machine Learning Model

The project uses **XGBoost**, a powerful gradient boosting algorithm.

The model was trained on **6500+ historical UFC fights**.

Performance:

-   Test Accuracy: \~71%
-   Cross Validation Accuracy: \~70%

------------------------------------------------------------------------

## Monte Carlo Fight Simulation

After predicting fight probabilities, the system simulates the fight
**10,000 times** using Monte Carlo simulation.

Example:

If the model predicts:

Holloway win probability: 0.61\
Oliveira win probability: 0.39

Then the simulator runs 10,000 fights and randomly samples winners based
on these probabilities.

------------------------------------------------------------------------

## Explainable AI (SHAP)

The project uses **SHAP** to explain why the model predicted a specific
outcome.

Example influential features:

-   ELO difference
-   Reach advantage
-   Striking volume
-   Win streak momentum

------------------------------------------------------------------------

# Project Structure

    ufc-xgboost-predictor

    data/
        ufc-master.csv
        max_charles_tonight.csv

    models/
        ufc_model.pkl
        feature_names.pkl

    src/
        train.py
        predict.py
        simulate.py
        explain.py
        elo.py

    requirements.txt
    README.md

------------------------------------------------------------------------

# Installation

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

# How to Run

Train the model:

    python src/train.py

Predict a fight:

    python src/predict.py

Run simulation:

    python src/simulate.py

Explain prediction:

    python src/explain.py

------------------------------------------------------------------------

# Technologies Used

-   Python
-   XGBoost
-   Scikit-Learn
-   Pandas
-   NumPy
-   SHAP
-   Monte Carlo Simulation

------------------------------------------------------------------------

# Example Prediction

Fight:

Max Holloway vs Charles Oliveira

Output:

Red win probability: 0.61\
Blue win probability: 0.39

------------------------------------------------------------------------

# Future Improvements

Possible upgrades:

-   Finish type prediction (KO, submission, decision)
-   Fighter style matchup modeling
-   Web dashboard for predictions
-   Real-time odds integration

------------------------------------------------------------------------

# Author

Machine learning project demonstrating sports analytics using UFC fight
data.
