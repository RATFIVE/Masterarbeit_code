import streamlit as st
import optuna
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import optuna.visualization as ov

# --- Helper function ---
def objective(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    model = RandomForestRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    return np.mean(scores)

# --- Streamlit UI ---
st.set_page_config(page_title="Optuna Dashboard", layout="wide")
st.title("Optuna-gestützte Random Forest Optimierung")

# Load dataset
dataset = fetch_california_housing()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar settings
df_placeholder = st.sidebar.empty()
n_trials = st.sidebar.number_input("Anzahl der Trials", min_value=5, max_value=500, value=5, step=5)
start_button = st.sidebar.button("Starte Optimierung")

# Initialize session state
if 'study' not in st.session_state:
    st.session_state.study = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'scores' not in st.session_state:
    st.session_state.scores = []

if start_button:
    # Create new study
    st.session_state.study = optuna.create_study(direction='maximize')
    st.session_state.progress = 0
    st.session_state.scores = []

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    chart_placeholder = st.sidebar.empty()

    # Run trials one by one to update chart
    for i in range(n_trials):
        st.session_state.study.optimize(
            lambda t: objective(t, X_train, y_train),
            n_trials=1
        )
        # Update progress and scores
        st.session_state.progress += 1
        latest_value = st.session_state.study.trials[-1].value # Get the latest trial value
        st.session_state.scores.append(latest_value)

        progress_bar.progress(st.session_state.progress / n_trials)
        status_text.text(f"Trial {st.session_state.progress} / {n_trials} (R²={latest_value:.4f})")
        # Update line chart
        scores_df = pd.DataFrame({'R2': st.session_state.scores}, index=range(1, len(st.session_state.scores)+1))
        chart_placeholder.line_chart(scores_df)

    st.success("Optimierung abgeschlossen!")

# Display results if study exists
if st.session_state.study is not None:
    study = st.session_state.study
    st.header("Optimierungsstatistiken")

    # Best params & score
    st.subheader("Bester Score & Beste Parameter")
    st.write(f"Best CV R² Score: {study.best_value:.4f}")
    st.write(study.best_params)
    

    # Visualization: history & importances
    st.subheader("Verlauf der Zielgröße")
    fig_hist = ov.plot_optimization_history(study)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Parameter-Importanzen")
    fig_imp = ov.plot_param_importances(study)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Parallel-Koordinaten-Plot")
    fig_parallel = ov.plot_parallel_coordinate(study)
    st.plotly_chart(fig_parallel, use_container_width=True)

    st.subheader("Slice-Plot")
    fig_slice = ov.plot_slice(study)
    st.plotly_chart(fig_slice, use_container_width=True)

    st.subheader("Contour-Plot")
    fig_contour = ov.plot_contour(study)
    st.plotly_chart(fig_contour, use_container_width=True)

    st.subheader("EDF-Plot")
    fig_edf = ov.plot_edf(study)
    st.plotly_chart(fig_edf, use_container_width=True)

    # Evaluate on test set
    st.header("Test-Set Evaluation")
    best_model = RandomForestRegressor(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    test_r2 = best_model.score(X_test, y_test)
    st.write(f"Test R² Score: {test_r2:.4f}")

    # CV Scores final model
    st.subheader("CV Scores finaler Modell (5-fach)")
    cvs=5
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cvs, scoring='r2', n_jobs=-1)
    df_scores = pd.DataFrame({"Fold": list(range(1,cvs+1)), "R2": cv_scores})
    fig_cv = px.bar(df_scores, x='Fold', y='R2', title='CV R² Scores')
    st.plotly_chart(fig_cv, use_container_width=True)