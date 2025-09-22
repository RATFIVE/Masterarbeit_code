# import all necessary libraries
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization as ov
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR
from tqdm import tqdm
from utils.config import (
    OCEAN_POINTS,
    WEATHER_POINTS,
)
from xgboost import XGBRegressor

# Ignore SettingWithCopyWarning:
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Display all columns
#pd.options.display.max_columns = None






plt.rcParams.update({
    "font.size": 14,                # Grundschriftgröße (wirkt auf alles, sofern nicht überschrieben)
    "axes.titlesize": 16,           # Größe des Titels der Achse (z.B. 'Subplot Title')
    "axes.labelsize": 14,           # Achsenbeschriftung (x/y label)
    "xtick.labelsize": 12,          # X-Tick-Beschriftung
    "ytick.labelsize": 12,          # Y-Tick-Beschriftung
    "legend.fontsize": 12,          # Legendentext
    "figure.titlesize": 18,         # Gesamttitel der Abbildung (plt.suptitle)
    "figure.labelsize": 14,         # (optional, selten verwendet)
    "savefig.dpi": 300,             # DPI beim Speichern
    "figure.dpi": 100,              # DPI bei Anzeige
})


ocean_data_path = Path(f"../data/numerical_data/points{OCEAN_POINTS}")
print(ocean_data_path)
weather_data_path = Path(f"../data/numerical_data/points{WEATHER_POINTS}")
print(weather_data_path)

def merge_dataframes(dfs: list) -> pd.DataFrame:
    """
    Merge multiple DataFrames on the 'time' column.
    """
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on='time', how='inner')
    return merged_df


def convert_df_to_table(df: pd.DataFrame) -> pd.DataFrame:


    df['position'] = df.apply(lambda row: (row['latitude'], row['longitude']), axis=1)
    coordinates = df['position'].unique()

    df_merged = pd.DataFrame({'time': df['time'].unique()})
    for i in tqdm(range(len(coordinates)), desc="Processing coordinates", unit="coord", total=len(coordinates)):

        df_sub_data = df[df['position'] == coordinates[i]]
        df_sub_data = df_sub_data.drop(columns=['latitude', 'longitude'])

        cols = df_sub_data.columns.tolist()
        cols.remove('position')
        cols.remove('time')


        for col in cols:
            df_sub_data.rename(columns={col: col + '_' + str(coordinates[i])}, inplace=True)

        df_sub_data = df_sub_data.drop(columns='position')


        df_merged = df_merged.merge(df_sub_data, on='time')
        
    return df_merged





# --- Model Optimizer Class ---
class ModelOptimizer:
    def __init__(self, model_name, param_space, X_train, y_train, cv=5, scoring='r2'):
        self.model_name = model_name
        self.param_space = param_space
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.scoring = scoring
        self.study = None

    def _suggest_params(self, trial):
        params = {}
        for name, cfg in self.param_space.items():
            ptype = cfg['type']
            if ptype == 'int':
                params[name] = trial.suggest_int(name, cfg['low'], cfg['high'])
            elif ptype == 'float':
                params[name] = trial.suggest_float(name, cfg['low'], cfg['high'], log=cfg.get('log', False))
            elif ptype == 'categorical':
                params[name] = trial.suggest_categorical(name, cfg['choices'])
            else:
                raise ValueError(f"Unknown parameter type {ptype} for {name}")
        return params

    def _create_model(self, params):
        if self.model_name == 'RandomForest':
            return RandomForestRegressor(**params, random_state=42)
        elif self.model_name == 'SVR':
            return SVR(**params)
        elif self.model_name == 'XGBRegressor':
            return XGBRegressor(**params, random_state=42)
        elif self.model_name == 'LinearRegression':
            return LinearRegression(**params)
        else:
            raise ValueError(f"Unknown model {self.model_name}")

    def objective(self, trial):
        params = self._suggest_params(trial)
        model = self._create_model(params)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv,
                                 scoring=self.scoring, n_jobs=-1)
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        return mean_score

    def optimize(self, n_trials):
            # Erstelle Study nur, wenn noch keins existiert
            if self.study is None:
                self.study = optuna.create_study(direction='maximize')
            # Führe n_trials aus, ergänze die bestehende study
            self.study.optimize(self.objective, n_trials=n_trials)
            return self.study
    

# --- Streamlit UI ---
st.set_page_config(page_title="Optuna Dashboard", layout="wide")
st.title("Optuna ML-Optimierung Dashboard")

# Load dataset
@st.cache_data
def load_data(real=False):
    
    if real:
        file_name = 'df_merged_all_FI.tsv'
        input_path = Path('../data/tabular_data_FI/')
        input_file = input_path / file_name
        df = pd.read_csv(input_file, sep='\t', parse_dates=['time'])

        X = df.drop(columns=["time", 'slev'])  # Features
        y = df['slev']  # Zielvariable
        st.dataframe(X.head())
        st.dataframe(y.head())
    else:
        # Load real dataset
        data = fetch_california_housing()
        X = data.data
        y = data.target

    return X, y

X, y = load_data(real=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Sidebar: Model selection
model_choice = st.sidebar.selectbox(
    "Wähle Modell:", ['RandomForest', 'SVR', 'XGBRegressor', 'LinearRegression']
)

# Define default parameter spaces
def get_param_space(model_name):
    if model_name == 'RandomForest':
        return {
            'n_estimators': {'type':'int','low':50,'high':300},
            'max_depth': {'type':'int','low':5,'high':50},
            'min_samples_split': {'type':'int','low':2,'high':20},
            'min_samples_leaf': {'type':'int','low':1,'high':20},
            'max_features': {'type':'categorical','choices':['sqrt', 'log2', None]},
            'bootstrap': {'type':'categorical','choices':[True, False]}
        }
    elif model_name == 'SVR':
        return {
            'C': {'type':'float','low':1e-3,'high':1e3,'log':True},
            'gamma': {'type':'float','low':1e-4,'high':1.0,'log':True},
            'epsilon': {'type':'float','low':1e-3,'high':1.0,'log':False},
            'kernel': {'type':'categorical','choices':['linear','rbf','poly']}
        }
    elif model_name == 'XGBRegressor':
        return {
            'n_estimators': {'type':'int','low':50,'high':300},
            'max_depth': {'type':'int','low':3,'high':15},
            'learning_rate': {'type':'float','low':0.01,'high':0.3,'log':True},
            'subsample': {'type':'float','low':0.1,'high':1.0},
            'colsample_bytree': {'type':'float','low':0.1,'high':1.0},
            'gamma': {'type':'float','low':0.0,'high':5.0}
        }
    else:
        return st.error("Model not supported")


    

param_space = get_param_space(model_choice)

# Sidebar: parameter ranges (optional adjustment)
st.sidebar.markdown("### Parameterbereich anpassen (optional)")
for name, cfg in param_space.items():
    if cfg['type'] in ['int', 'float']:
        low, high = cfg['low'], cfg['high']
        if cfg['type']=='int':
            new_low = st.sidebar.number_input(f"{name} min", value=low, step=1)
            new_high = st.sidebar.number_input(f"{name} max", value=high, step=1)
        else:
            new_low = st.sidebar.number_input(f"{name} min", value=low, format="%f")
            new_high = st.sidebar.number_input(f"{name} max", value=high, format="%f")
        cfg['low'], cfg['high'] = new_low, new_high
    else:
        # categorical leave as default or add multi-select
        pass

# Sidebar: trials & start
n_trials = st.sidebar.number_input("Anzahl der Trials", min_value=5, max_value=500, value=5, step=5)
start_button = st.sidebar.button("Starte Optimierung")

# Session state init
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'study' not in st.session_state:
    st.session_state.study = None


# Initialize optimizer
if st.session_state.optimizer is None:
    st.session_state.optimizer = ModelOptimizer(model_choice, param_space, X_train, y_train)
# Run optimization
if start_button:
    st.session_state.progress = 0
    st.session_state.scores = []

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    chart_placeholder = st.sidebar.empty()

    for i in range(n_trials):
        st.session_state.study = st.session_state.optimizer.optimize(n_trials=1)
        st.session_state.progress += 1
        val = st.session_state.study.trials[-1].value
        st.session_state.scores.append(val)
        progress_bar.progress(st.session_state.progress / n_trials)
        status_text.text(f"Trial {st.session_state.progress}/{n_trials} (Score={val:.4f})")
        df = pd.DataFrame({'Score': st.session_state.scores}, index=range(1,len(st.session_state.scores)+1))
        chart_placeholder.line_chart(df)
    st.success("Optimierung abgeschlossen!")

# Display results
if st.session_state.optimizer:
    study = st.session_state.study
    st.header("Ergebnisse")
    st.subheader("Bester Score & Parameter")
    if study is not None:
        st.write(f"Best Score: {study.best_value:.4f}")
        st.write(study.best_params)

        # Plots
        st.subheader("Optimierungsverlauf")
        st.plotly_chart(ov.plot_optimization_history(study), use_container_width=True)
        st.subheader("Parameter-Importanzen")
        st.plotly_chart(ov.plot_param_importances(study), use_container_width=True)
        st.subheader("Parallele Koordinaten")
        st.plotly_chart(ov.plot_parallel_coordinate(study), use_container_width=True)
        st.subheader("Slice-Plot")
        st.plotly_chart(ov.plot_slice(study), use_container_width=True)
        st.subheader("Kontur-Plot")
        st.plotly_chart(ov.plot_contour(study), use_container_width=True)
        st.subheader("EDF-Plot")
        st.plotly_chart(ov.plot_edf(study), use_container_width=True)

        # Final model evaluation
        st.header("Test-Set Auswertung")
        best_params = study.best_params
        if model_choice == 'RandomForest':
            final_model = RandomForestRegressor(**best_params, random_state=42)
        elif model_choice == 'XGBRegressor':
            final_model = XGBRegressor(**best_params, random_state=42)
        elif model_choice == 'SVR':
            final_model = SVR(**best_params)
        else:
            final_model = LinearRegression(**best_params)
        final_model.fit(X_train, y_train)
        test_score = final_model.score(X_test, y_test)
        st.write(f"Test Score: {test_score:.4f}")

        # CV Scores
        st.subheader("CV Scores des finalen Modells")
        cvs = st.sidebar.slider("CV Folds", 2, 10, 10)
        cv_scores = cross_val_score(final_model, X_train, y_train, cv=cvs, scoring='r2', n_jobs=-1)
        df_cv = pd.DataFrame({"Fold": list(range(1,cvs+1)), "Score": cv_scores})
        fig = px.bar(df_cv, x='Fold', y='Score', title='CV Scores')
        st.plotly_chart(fig, use_container_width=True)
