
# UFC Fight Predictor using Machine Learning 🥊

This project is an **end‑to‑end Machine Learning pipeline** designed to predict the outcome of UFC fights.  
It uses historical fight data, engineered fighter metrics, dynamic ELO ratings, and Monte Carlo simulations to estimate win probabilities.

The system was recently updated after testing on **Holloway vs Oliveira (UFC 326)** and retrained on the latest dataset.

---

# Project Overview

The model analyzes **6,500+ historical UFC fights** and learns patterns that influence fight outcomes.

Key factors considered:

- **Striking volume** – Significant strikes landed per minute
- **Grappling pressure** – Takedowns and submission attempts
- **Physical advantages** – Height, reach, and age
- **Fighter momentum** – Win streaks and losing streaks
- **Strength of schedule** – Opponent quality using ELO ratings

---

# Latest Model Results

Dataset size: **6,529 fights**  
Features used: **100**  
Model: **XGBoost Gradient Boosted Trees**

Test Accuracy:

**70.06%**

This indicates the model is learning meaningful patterns from historical UFC fight data.

---

# Example Prediction

Fight simulated: **Max Holloway vs Charles Oliveira**

Model prediction:

Max Holloway (Red Corner): **53% win probability**  
Charles Oliveira (Blue Corner): **47% win probability**

---

# Monte Carlo Fight Simulation

To account for performance variability, the system runs **10,000 simulated fights** with random performance noise.

Simulation Results:

Max Holloway win rate: **53.31%**  
Charles Oliveira win rate: **46.69%**

This indicates a **competitive matchup**, with Holloway slightly favored by the model.

---

# Key Machine Learning Features

## Grappling Pressure Feature

To capture grappling dominance, a custom metric was introduced:

GrapplingScore = AvgTDLanded + (AvgSubAtt × 2)

Submission attempts are weighted more heavily to reflect their higher fight‑ending potential.

---

## Dynamic ELO Rating System

Fighters receive ratings similar to chess ELO systems.

- Fighters start around **1500 ELO**
- Ratings increase after wins and decrease after losses
- Defeating stronger opponents produces larger rating gains

This helps model **strength of schedule** and overall fighter skill.

---

# Monte Carlo Simulation

Instead of predicting only one probability, the system simulates thousands of fights by adding small random variations to fighter performance metrics.

This provides:

- realistic variance
- probabilistic outcomes
- fight consistency evaluation

---

# Project Structure

```
ufc-fight-predictor/
│
├── data/
│   ├── ufc-master.csv
│   └── max_charles_tonight.csv
│
├── models/
│   ├── ufc_model.pkl
│   ├── feature_names.pkl
│   └── top_features.pkl
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── simulate.py
│   ├── elo.py
│   ├── update_data.py
│   └── explain.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

# Usage

### Update dataset with new fight results

```
python src/update_data.py
```

### Train the model

```
python src/train.py
```

### Predict a fight

```
python src/predict.py
```

### Run Monte Carlo simulations

```
python src/simulate.py
```

### Explain prediction with SHAP

```
python src/explain.py
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- XGBoost
- Scikit‑Learn
- SHAP
- Monte Carlo Simulation

---

# Author

**Kyaw Soe Lwin**  
Data Science Student  
Skyline College & College of San Mateo  

Interested in **Machine Learning, AI Engineering, and Sports Analytics**
