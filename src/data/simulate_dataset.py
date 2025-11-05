"""
Simulate a student dataset and save to `data/raw/students.csv`.
Features include demographics, academic performance, engagement, socio-economic indicators, and a binary dropout target.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RND = 42
np.random.seed(RND)

OUT = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

n = 5000  # number of students

# Demographics
age = np.random.randint(15, 22, size=n)
gender = np.random.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.5, 0.02])
parent_income = np.random.normal(40000, 15000, size=n).clip(5000, 200000)

# Academic performance
gpa = np.round(np.random.normal(2.8, 0.6, size=n), 2).clip(0.0, 4.0)
attendance_pct = np.round(np.random.normal(88, 10, size=n), 1).clip(30, 100)
midterm_score = np.round(np.random.normal(68, 15, size=n)).clip(0, 100)
final_score = np.round(midterm_score + np.random.normal(5, 10, size=n)).clip(0, 100)

# Engagement
assign_submission_rate = np.round(np.random.normal(0.85, 0.2, size=n), 2).clip(0, 1)
lms_logins_week = np.random.poisson(3, size=n)
participation_score = np.round(np.random.normal(0.6, 0.25, size=n), 2).clip(0, 1)

# Socio-economic
household_size = np.random.choice([1,2,3,4,5,6], size=n, p=[0.05,0.2,0.3,0.25,0.15,0.05])
parent_education = np.random.choice(["high_school","bachelors","masters","none"], size=n, p=[0.4,0.35,0.2,0.05])
urban = np.random.choice([0,1], size=n, p=[0.3,0.7])

# Create dataframe
df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "parent_income": parent_income,
    "gpa": gpa,
    "attendance_pct": attendance_pct,
    "midterm_score": midterm_score,
    "final_score": final_score,
    "assign_submission_rate": assign_submission_rate,
    "lms_logins_week": lms_logins_week,
    "participation_score": participation_score,
    "household_size": household_size,
    "parent_education": parent_education,
    "urban": urban,
})

# Generate dropout with a logistic combination of risk factors
# Students with low GPA, low attendance, low submissions, low income have higher dropout risk
logit = (
    -2.5
    + (2.0 - df["gpa"]) * 1.2
    + (75 - df["attendance_pct"]) * 0.03
    + (1 - df["assign_submission_rate"]) * 1.5
    + (30000 - df["parent_income"]) / 50000
    - (df["participation_score"] - 0.4) * 1.0
)
prob = 1 / (1 + np.exp(-logit))
dropout = np.where(np.random.rand(n) < prob, "Yes", "No")

df["dropout"] = dropout

# Introduce some missingness randomly
for col in ["parent_income", "gpa", "attendance_pct", "assign_submission_rate"]:
    mask = np.random.rand(n) < 0.02
    df.loc[mask, col] = np.nan

# Save CSV
out_path = OUT / "students.csv"
df.to_csv(out_path, index=False)
print(f"Saved simulated dataset to {out_path}")
