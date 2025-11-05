# Predicting Student Dropout using Machine Learning

This project demonstrates a complete end-to-end data science and web application pipeline for predicting student dropout risk. It includes dataset simulation, preprocessing, model training and evaluation, and an interactive Streamlit dashboard for predictions and analytics.

## 1 — Problem definition & overview

Predicting student dropout means using student data (demographics, academic performance, engagement, and socio-economic indicators) to estimate whether a student is at risk of leaving school before completion. Early identification helps institutions intervene with targeted support (tutoring, counseling, financial aid), improving retention rates and student outcomes.

Objectives
- Build a machine learning pipeline to predict dropout (binary: Yes/No).
- Compare multiple models and choose a best-performing model with explainability.
- Provide an interactive front-end for administrators to predict and explore insights.

Expected outcomes
- A reproducible dataset generation and preprocessing pipeline.
- Trained and evaluated models (Logistic Regression, Random Forest, XGBoost, SVM, MLP).
- Visualizations for model performance and feature importance.
- A Streamlit app to input new student data and view analytics.

## 2 — Project structure

- `data/` — raw and processed datasets
- `notebooks/` — exploratory analysis and EDA
- `src/data/` — data generation and preprocessing scripts
- `src/models/` — training and utilities
- `src/app/` — Streamlit frontend
- `outputs/` — trained models, figures, and metrics

## 3 — Quick start (development)

1. Create and activate a Python environment (Python 3.8+ recommended).
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Generate the synthetic dataset:

```powershell
python src/data/simulate_dataset.py
```

4. Prepare the data (cleaning, encoding, scaling):

```powershell
python src/data/prepare_data.py
```

5. Train models and produce evaluation outputs:

```powershell
python src/models/train_models.py
```

6. Run the Streamlit app:

```powershell
streamlit run src/app/streamlit_app.py
```

## 4 — Notes

- This repository uses a simulated dataset for reproducibility. Replace `data/raw/students.csv` with a real dataset when available. Ensure proper privacy handling when using real student records.
- The Streamlit app is intended for demos. For production deployment, consider containerization and secure model-serving.

---

See `requirements.txt` for exact packages and versions.
