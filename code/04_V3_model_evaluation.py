
# # Model Evaluation


# ## Libaries



# Drittanbieter
import time
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.svm import SVR
from utils.config import PRED_COLOR, TRUE_COLOR
from utils.dl_helper_functions import (
    load_picture_lagged_data,
)

# Eigene Module (utils)
from xgboost import XGBRegressor

DTYPE_NUMPY = np.float32
SEQUENCE_LENGTH = 24  # Anzahl der Schritte in der Sequenz
n_jobs = 16


import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.multioutput import MultiOutputRegressor
from utils.dl_helper_functions import (
    create_sequences,
    scale_data,
)

# # Versuch 3 ML


# ## Class



# === SETTINGS ===
HORIZON = 24
SEQUENCE_LENGTH = 24
DTYPE_NUMPY = np.float32
n_jobs = -1
model_name = "XGBoost"  # ⬅ ändere hier je nach Modell
# models = ["RandomForest", "XGBoost", "SVR", "LGBM", "Linear"]





# === Load data ===
X, y_lagged, y, common_time = load_picture_lagged_data(
    return_common_time=True,
    verbose=False,
    grid_size=25,
    n_jobs=n_jobs,
    dtype=DTYPE_NUMPY,
    pca=True,
)

X = X.astype(DTYPE_NUMPY)
y_lagged = y_lagged.astype(DTYPE_NUMPY)
y = y.astype(DTYPE_NUMPY)





class MLModelEvaluator:
    def __init__(self, X, y_lagged, y, common_time, sequence_length, horizon, n_jobs, dtype=np.float32, model_name="XGBoost"):
        self.X = X.astype(dtype)
        self.y_lagged = y_lagged.astype(dtype)
        self.y = y.astype(dtype)
        self.common_time = common_time
        self.SEQUENCE_LENGTH = sequence_length
        self.HORIZON = horizon
        self.n_jobs = n_jobs
        self.results_all_models = []
        self.model_name = model_name
        self.folds = {
                "Surge1": pd.Timestamp("2023-02-25 16:00:00"),
                "Surge2": pd.Timestamp("2023-04-01 09:00:00"),
                "Surge3": pd.Timestamp("2023-10-07 20:00:00"),
                "Surge4": pd.Timestamp("2023-10-20 21:00:00"),
                "Surge5": pd.Timestamp("2024-01-03 01:00:00"),
                "Surge6": pd.Timestamp("2024-02-09 15:00:00"),
                "Surge7": pd.Timestamp("2024-12-09 10:00:00"),
                "normal1": pd.Timestamp("2023-07-01 14:00:00"),
                "normal2": pd.Timestamp("2024-04-01 18:00:00"),
                "normal3": pd.Timestamp("2025-01-01 12:00:00"),
                }

    def get_model(self, name, trial_params):
        if name == "RandomForest":
            return MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=self.n_jobs, **trial_params), n_jobs=self.n_jobs)
        elif name == "SVR":
            return MultiOutputRegressor(SVR(**trial_params), n_jobs=self.n_jobs)
        elif name == "XGBoost":
            return MultiOutputRegressor(XGBRegressor(random_state=42, n_jobs=self.n_jobs, **trial_params), n_jobs=self.n_jobs)
        elif name == "LGBM":
            return MultiOutputRegressor(LGBMRegressor(random_state=42, n_jobs=self.n_jobs, **trial_params), n_jobs=self.n_jobs)
        elif name == "Linear":
            return MultiOutputRegressor(LinearRegression(n_jobs=self.n_jobs), n_jobs=self.n_jobs)
        else:
            raise ValueError(f"Unbekanntes Modell: {name}")

    def custom_score(self, y_true, y_pred, bins=[1, 1.25, 1.5, 2.0], alpha=0.7):
        recalls = [recall_score(np.digitize(y_true[:, i], bins), np.digitize(y_pred[:, i], bins), average="macro") for i in range(y_true.shape[1])]
        mean_recall = np.mean(recalls)
        mse = mean_squared_error(y_true, y_pred)
        return alpha * (1 - mean_recall) + (1 - alpha) * mse

    def recall_per_fold(self, y_true, y_pred, bins=[1, 1.25, 1.5, 2.0]):
        recalls = [recall_score(np.digitize(y_true[:, i], bins), np.digitize(y_pred[:, i], bins), average="macro") for i in range(y_true.shape[1])]
        return np.mean(recalls)

    def plot_last_forecast(self, model, X_test, y_lagged_test, y_test, time_index):
        # Vorbereitung
        X_test_flat = np.hstack([X_test.reshape(X_test.shape[0], -1), y_lagged_test.reshape(y_lagged_test.shape[0], -1)])
        y_pred = model.predict(X_test_flat)  # Erwartet: shape (num_blocks, HORIZON)
        mse = mean_squared_error(y_test, y_pred)

        # Sicherstellen, dass time_index ein DatetimeIndex ist und lang genug
        if not isinstance(time_index, (pd.DatetimeIndex, pd.core.indexes.datetimes.DatetimeIndex)):
            time_index = pd.to_datetime(time_index)
        num_blocks = y_test.shape[0]
        required_length = num_blocks + self.HORIZON  # damit auch das letzte Forecast-Fenster abgedeckt ist
        if len(time_index) < required_length:
            raise ValueError(f"time_index zu kurz: braucht mindestens {required_length}, hat aber {len(time_index)}")

        # Highlight storm surge classes
        flood_levels = [
            (1.0, 1.25, 'yellow', 'storm surge'),
            (1.25, 1.5, 'orange', 'medium storm surge'),
            (1.5, 2.0, 'red', 'heavy storm surge'),
            (2.0, 3.5, 'darkred', 'very heavy storm surge'),
        ]

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plotte die Forecast-Blöcke mit echten Zeitstempeln
        step = self.HORIZON  # z.B. 24, wenn alle 24h ein neuer Forecast startet
        for i in range(0, num_blocks, step):
            y_pred_block = y_pred[i]  # Vektor der Länge HORIZON
            y_true_block = y_test[i]
            x_block = time_index[i : i + self.HORIZON]

            label_pred = 'Forecasts' if i == 0 else None
            label_true = 'True Values' if i == 0 else None
            ax.plot(x_block, y_pred_block, color=PRED_COLOR, linestyle='-', label=label_pred, alpha=0.7)
            ax.plot(x_block, y_true_block, color=TRUE_COLOR, linestyle='-', label=label_true, alpha=0.7)
            if i == 0:
                ax.axvline(x=time_index[i], color='gray', linestyle=':', alpha=0.5, label='Begin New Forecast')
            else:
                ax.axvline(x=time_index[i], color='gray', linestyle=':', alpha=0.5)

        # Flood level Hintergrund
        for y0, y1, color, label in flood_levels:
            ax.axhspan(y0, y1, facecolor=color, alpha=0.2, label=label)

        # Vertikale Linien für jeden Forecast-Beginn (falls nicht schon durch obige Schleife ausreichend)
        for i in range(0, num_blocks, step):
            ax.axvline(x=time_index[i], color='gray', linestyle='--', alpha=0.1)

        # Null-Linie
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # Metrik einblenden
        # ax.text(0.99, 0.02, f"MSE: {mse:.4f}", horizontalalignment='right',
        #         verticalalignment='bottom', transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

        # Achsen & Titel
        ax.set_xlabel("Time")
        ax.set_ylabel("Water Level [m]")
        ax.set_title("Forecast vs. Actual Water Level")

        # y-Limits
        y_min = min(y_test.min(), y_pred.min())
        y_max = max(y_test.max(), y_pred.max())
        ax.set_ylim(y_min - 0.5, y_max + 0.5)

        # Formatierung
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        fig.autofmt_xdate()

        # Legende deduplizieren
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3)

        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"../thesis_plots/Versuch3_forecast_{self.model_name}_{time_index[0].strftime('%Y%m%d_%H%M')}.png", dpi=300)
        #plt.show()

    def plot_results(self, scores_df):
        for metric in ["RMSE"]:
            plt.figure(figsize=(10, 6))
            plt.bar(scores_df["fold"], scores_df[metric], color=TRUE_COLOR)
            mean_value = scores_df[metric].mean()
            std_value = scores_df[metric].std()
            
            plt.axhspan(mean_value - std_value, mean_value + std_value, 
                        color='green', alpha=0.2, 
                        label=f'±1 STD: {std_value:.4f}')
            plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean {metric.upper()}: {mean_value:.4f}')
            plt.xlabel("Fold")
            plt.ylabel(metric.upper())
            plt.legend()
            plt.title(f"{self.model_name} Cross-Validation {metric.upper()} (Best Parameters)")
            plt.xticks(rotation=45)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"../thesis_plots/Versuch3_{self.model_name}_cross_validation_{metric.lower()}.png", dpi=300)
            #plt.show()

    def evaluate_model(self, best_params, create_sequences, scale_data):
        
        prediction_df = pd.DataFrame(columns=["model_name", "y_true", "y_pred"])
        results = []

        for surge_name, fold in self.folds.items():
            start_cutoff = fold - pd.Timedelta(hours=168 * 4)
            end_cutoff = fold + pd.Timedelta(hours=168 * 4)
            idx_start = np.where(self.common_time == start_cutoff)[0][0]
            idx_end = np.where(self.common_time == end_cutoff)[0][0]

            X_test, y_lagged_test, y_test = self.X[idx_start:idx_end], self.y_lagged[idx_start:idx_end], self.y[idx_start:idx_end]
            X_train, y_lagged_train, y_train = self.X.copy(), self.y_lagged.copy(), self.y.copy()
            X_train[idx_start:idx_end], y_lagged_train[idx_start:idx_end], y_train[idx_start:idx_end] = np.nan, np.nan, np.nan

            X_train, y_lagged_train, y_train = create_sequences(X_train, y_lagged_train, y_train, self.SEQUENCE_LENGTH, self.HORIZON)
            X_test, y_lagged_test, y_test = create_sequences(X_test, y_lagged_test, y_test, self.SEQUENCE_LENGTH, self.HORIZON)

            gap = 168
            X_test, y_lagged_test, y_test = X_test[gap:-gap], y_lagged_test[gap:-gap], y_test[gap:-gap]

            X_train, y_lagged_train, y_train, _, _, _, X_test, y_lagged_test, y_test, _, _ = scale_data(
                X_train, y_lagged_train, y_train, None, None, None, X_test, y_lagged_test, y_test, dtype=self.X.dtype, verbose=False
            )

            X_train_flat = np.hstack([X_train.reshape(X_train.shape[0], -1), y_lagged_train.reshape(y_lagged_train.shape[0], -1)])
            X_test_flat = np.hstack([X_test.reshape(X_test.shape[0], -1), y_lagged_test.reshape(y_lagged_test.shape[0], -1)])

            # check if X_train_flat, y_train, X_test_flat are Pandas DataFrames
            if isinstance(X_train_flat, pd.DataFrame):
                X_train_flat = X_train_flat.values
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.values
            if isinstance(X_test_flat, pd.DataFrame):
                X_test_flat = X_test_flat.values
            model = self.get_model(self.model_name, best_params)
            fit_time_start = time.time()
            model.fit(X_train_flat, y_train)
            fit_time_end = time.time()
            fit_time = fit_time_end - fit_time_start

            predict_time_start = time.time()
            y_pred = model.predict(X_test_flat)
            predict_time_end = time.time()
            predict_time = predict_time_end - predict_time_start

            score = self.custom_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            recall = self.recall_per_fold(y_test, y_pred)

            print(f"{surge_name}: Score = {score:.4f}, MSE = {mse:.4f}, Recall = {recall:.4f}")
            time_index = common_time[idx_start + gap : idx_end - gap]
            self.plot_last_forecast(model, X_test, y_lagged_test, y_test, time_index)

            results.append({"fold": surge_name, 
                            "SCORE": score, 
                            "MSE": mse, 
                            "RMSE": rmse, 
                            "RECALL": recall,
                            "fit_time": fit_time,
                            "predict_time": predict_time,
                            "Num Params": self.count_params(model),
                            })
            prediction_df = pd.concat([prediction_df, pd.DataFrame({
                "model_name": self.model_name,
                "y_true": y_test.flatten(),
                "y_pred": y_pred.flatten(),
                
            })], ignore_index=True)

        scores_df = pd.DataFrame(results)
        print("\n=== Cross-Validation Results ===")
        print(scores_df)
        print("\nMean Score:", scores_df["SCORE"].mean())

        self.plot_results(scores_df)

        return prediction_df

    def get_best_params(self, horizon):
        if self.model_name != "Linear":
            storage = f"sqlite:///Versuch3_{self.model_name}.db"
            study = optuna.load_study(study_name=f"{horizon}", storage=storage)
            return study.best_params
        else:
            return {}
    def count_params(self, model):
        import lightgbm as lgb
        import xgboost as xgb
        # Falls MultiOutputRegressor: aufsummieren
        if hasattr(model, "estimators_") and isinstance(model, (MultiOutputRegressor,)):
            return sum(self.count_params(est) for est in model.estimators_)

        # Linear Regression: Koeffizienten + Intercept
        from sklearn.linear_model import LinearRegression
        if isinstance(model, LinearRegression):
            coef_count = np.prod(model.coef_.shape)
            intercept_count = np.prod(np.atleast_1d(model.intercept_).shape)
            return int(coef_count + intercept_count)

        # RandomForest / DecisionTree: Anzahl Knoten in den Bäumen
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.tree import DecisionTreeRegressor
        if isinstance(model, DecisionTreeRegressor):
            return int(model.tree_.node_count)
        if isinstance(model, RandomForestRegressor):
            return sum(int(est.tree_.node_count) for est in model.estimators_)

        # SVR: Anzahl Support-Vektoren
        from sklearn.svm import SVR
        if isinstance(model, SVR):
            # support_ ist ein Array pro Ziel (bei MultiOutputRegressor wäre oben schon abgefangen)
            return int(np.sum([len(model.support_)]))  # schlicht: Anzahl Support-Vektoren

        # XGBoost
        if isinstance(model, xgb.XGBRegressor):
            booster = model.get_booster()
            dump = booster.get_dump(with_stats=False)  # Liste pro Baum
            # einfache Proxy: Anzahl Zeilen (Knoten) über alle Bäume summieren
            return sum(t.count("\n") + 1 for t in dump)

        # LightGBM
        if isinstance(model, lgb.LGBMRegressor):
            booster = model.booster_
            model_str = booster.model_to_string()
            # Einfacher Proxy: Anzahl Trees
            try:
                num_trees = booster.num_trees()
                return int(num_trees)
            except:
                return np.nan

        # Fallback
        return np.nan


def get_model_statistics(model_name, results_df=None):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        mean_squared_error,
        recall_score,
    )

    """
    Gibt ein DataFrame mit den wichtigsten Modellstatistiken zurück.
    """
    if results_df is None or results_df.empty:
        results_df = pd.DataFrame()


    #group results_df by model name
    df_grouped = results_df.groupby("model_name").get_group(model_name)

    bins=[1, 1.25, 1.5, 2.0]

    y_true = df_grouped["y_true"].values
    y_pred = df_grouped["y_pred"].values

    y_true_class = np.digitize(y_true, bins)
    y_pred_class = np.digitize(y_pred, bins)

    # Fehlermaße
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Korrelation
    try:
        r_value, _ = pearsonr(y_true, y_pred)
        r_squared = r2_score(y_true, y_pred)
    except:
        r_value = np.nan



    recall = recall_score(y_true_class, y_pred_class, average="macro")
    precision = precision_score(y_true_class, y_pred_class, average="macro")
    accuracy = accuracy_score(y_true_class, y_pred_class)
    f1 = f1_score(y_true_class, y_pred_class, average="macro")

    

    # Laufzeit-Mittelwerte (falls vorhanden)
    if 'fit_time' in results_df.columns and 'predict_time' in results_df.columns:
        avg_fit_time = results_df['fit_time'].mean()
        avg_predict_time = results_df['predict_time'].mean()
    else:
        avg_fit_time = np.nan
        avg_predict_time = np.nan

    
        
    # Zusammenfassung
    stats = {
        'Model': model_name,
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        #"R": round(r_value, 3),
        #"R²": round(r_squared, 3),
        "Recall": round(recall, 3),
        "Precision": round(precision, 3),
        "Accuracy": round(accuracy, 3),
        "F1-Score": round(f1, 3),
        "Avg. Fit Time (s)": round(avg_fit_time, 3),
        "Avg. Predict Time (s)": round(avg_predict_time, 3),
        #"Num Params": num_params
    }

    return pd.DataFrame([stats])




results_df = pd.DataFrame()


# ## XGBoost


# Initialisiere Evaluator 
evaluator = MLModelEvaluator(
    X=X,
    y_lagged=y_lagged,
    y=y,
    common_time=common_time,
    sequence_length=SEQUENCE_LENGTH,
    horizon=HORIZON,
    n_jobs=n_jobs,
    dtype=DTYPE_NUMPY,
    model_name="XGBoost"  # Oder "RandomForest", "SVR", "LGBM", "Linear",
)
# Best Parameters
best_params = evaluator.get_best_params(HORIZON)

# Evaluation 
prediction_df = evaluator.evaluate_model(
    best_params=best_params,
    create_sequences=create_sequences,
    scale_data=scale_data,
)

results_df = pd.concat([results_df, get_model_statistics("XGBoost", prediction_df)], ignore_index=True)
#display(results_df)


# ## Linear Regression


# Initialisiere Evaluator 
evaluator = MLModelEvaluator(
    X=X,
    y_lagged=y_lagged,
    y=y,
    common_time=common_time,
    sequence_length=SEQUENCE_LENGTH,
    horizon=HORIZON,
    n_jobs=n_jobs,
    dtype=DTYPE_NUMPY,
    model_name="Linear"  # Oder "RandomForest", "SVR", "LGBM", "Linear",
)
# Best Parameters
best_params = evaluator.get_best_params(HORIZON)

# Evaluation 
prediction_df = evaluator.evaluate_model(
    best_params=best_params,
    create_sequences=create_sequences,
    scale_data=scale_data,
)

results_df = pd.concat([results_df, get_model_statistics("Linear", prediction_df)], ignore_index=True)
#display(results_df)


# ## Random Forest


# Initialisiere Evaluator 
evaluator = MLModelEvaluator(
    X=X,
    y_lagged=y_lagged,
    y=y,
    common_time=common_time,
    sequence_length=SEQUENCE_LENGTH,
    horizon=HORIZON,
    n_jobs=n_jobs,
    dtype=DTYPE_NUMPY,
    model_name="RandomForest"  # Oder "RandomForest", "SVR", "LGBM", "Linear",
)
# Best Parameters
best_params = evaluator.get_best_params(HORIZON)

# Evaluation 
prediction_df = evaluator.evaluate_model(
    best_params=best_params,
    create_sequences=create_sequences,
    scale_data=scale_data,
)

results_df = pd.concat([results_df, get_model_statistics("RandomForest", prediction_df)], ignore_index=True)
#display(results_df)


# ## SVR


# Initialisiere Evaluator 
evaluator = MLModelEvaluator(
    X=X,
    y_lagged=y_lagged,
    y=y,
    common_time=common_time,
    sequence_length=SEQUENCE_LENGTH,
    horizon=HORIZON,
    n_jobs=n_jobs,
    dtype=DTYPE_NUMPY,
    model_name="SVR"  # Oder "RandomForest", "SVR", "LGBM", "Linear",
)
# Best Parameters
best_params = evaluator.get_best_params(HORIZON)

# Evaluation 
prediction_df = evaluator.evaluate_model(
    best_params=best_params,
    create_sequences=create_sequences,
    scale_data=scale_data,
)

results_df = pd.concat([results_df, get_model_statistics("SVR", prediction_df)], ignore_index=True)
#display(results_df)


# ## LGBM


# Initialisiere Evaluator 
evaluator = MLModelEvaluator(
    X=X,
    y_lagged=y_lagged,
    y=y,
    common_time=common_time,
    sequence_length=SEQUENCE_LENGTH,
    horizon=HORIZON,
    n_jobs=n_jobs,
    dtype=DTYPE_NUMPY,
    model_name="LGBM"  # Oder "RandomForest", "SVR", "LGBM", "Linear",
)
# Best Parameters
best_params = evaluator.get_best_params(HORIZON)

# Evaluation 
prediction_df = evaluator.evaluate_model(
    best_params=best_params,
    create_sequences=create_sequences,
    scale_data=scale_data,
)

results_df = pd.concat([results_df, get_model_statistics("LGBM", prediction_df)], ignore_index=True)
#display(results_df)
results_df.to_csv("../thesis_plots/Versuch3_ML_model_statistics_all_models.csv", index=False)



