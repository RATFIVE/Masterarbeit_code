
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
from mlforecast import MLForecast
from mlforecast.feature_engineering import transform_exog
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
from utils.dl_helper_functions import (
    load_picture_lagged_data,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from statsforecast import StatsForecast
from statsforecast.models import ARIMA
from tqdm import tqdm
from utils.config import (PRED_COLOR, TRUE_COLOR)

# Eigene Module (utils)
from utils.ml_helper_functions import load_data, load_data_v2
from xgboost import XGBRegressor
DTYPE_NUMPY = np.float32
SEQUENCE_LENGTH = 24  # Anzahl der Schritte in der Sequenz
n_jobs = -1


# # Versuch 1


# ## Load Data


df = load_data()
#df = load_data_v2()


df


# ## Database files


database_files = ['optuna_study_ML_xgboost.db', 'optuna_study_ML_lgbm.db', 'optuna_study_ML_lr.db', 'optuna_study_ML_rf.db', 'optuna_study_ML_svr.db', 'optuna_study_ML_sarima.db']





class EvaluationMLModel:
    def __init__(self, model_type, df, h=24, ocean_points=30, Versuch: str = "1", trial_name: str = "default"):

        self.trial_name = trial_name
        
        self.model_type = model_type
        self.h = h
        self.ocean_points = ocean_points
       
        if self.model_type == 'sarima':
            self.db_name = f'Versuch{Versuch}_optuna_study_ML_sarima.db'
        elif self.model_type == 'xgboost':
            self.db_name = f'Versuch{Versuch}_optuna_study_ML_xgboost.db'
        elif self.model_type == 'lgbm':
            self.db_name = f'Versuch{Versuch}_optuna_study_ML_lgbm.db'
        elif self.model_type == 'random_forest':
            self.db_name = f'Versuch{Versuch}_optuna_study_ML_rf.db'
        elif self.model_type == 'svr':
            self.db_name = f'Versuch{Versuch}_optuna_study_ML_svr.db'
        elif self.model_type == 'lr':
            self.db_name = f'Versuch{Versuch}_optuna_study_ML_lr.db'
        else:
            raise ValueError("Unbekannter Modelltyp. Bitte 'sarima', 'xgboost', 'lgbm', 'random_forest', 'svr' oder 'lr' verwenden.")
        self.n_jobs = n_jobs  # Alle verfügbaren CPU-Kerne nutzen
        self.study = self._open_study_db(self.db_name)
        self.best_params = self._best_params(self.study)
        self.lags = self.best_params.pop('lags')
        self.model = self._model()
        self.df_all = None

        # Transformiere die Exogenen Variablen und entferne die Lags-Spalten
        self.df = transform_exog(df, lags=self.lags, num_threads=-1)     
        cols_to_remove = [col for col in self.df.columns if col.startswith('y_lag')]
        self.df = self.df.drop(columns=set(cols_to_remove))

        

        self.df_train = self.df.loc[self.df['ds'] < '2025-03-01']
        self.df_temp = self.df.loc[self.df['ds'] >= '2025-03-01']
        self.df_val = self.df_temp.iloc[:-self.h]
        self.df_test = self.df_temp.iloc[-self.h:]

        # 
        if self.model_type == "sarima":
            # Use just the columns ['ds', 'unique_id', 'y'] for SARIMA
            self.df_train = self.df_train[["ds", "unique_id", "y"]]
            self.df_temp = self.df_temp[["ds", "unique_id", "y"]]
            self.df_val = self.df_val[["ds", "unique_id", "y"]]
            self.df_test = self.df_test[["ds", "unique_id", "y"]]

        self.random_state = 42  # Für Reproduzierbarkeit
        

        # Parametriere die Cross-Validation-Window-Logik
        self.len_df = len(self.df)
        self.initial_train_window = 24 * 30 * 6   # 6 Monate
        self.step_size = 24                   # 7 Tage
        self.horizon = 24                    # 7 Tage
        self.n_windows = ((self.len_df - self.initial_train_window) // self.horizon) + 200
        self._print_model_info()

    def _print_model_info(self):
        ##display(self.df.head())
        print(self.df.columns.to_list())
        print(self.df.info())
        print(f"Modelltyp: {self.model_type}")
        print(f"Optimierungsstudie: {self.db_name}")
        print(f"Beste Parameter: {self.best_params}")
        print(f"Lags: {self.lags}")
        print(f"Trainingsdaten: {len(self.df_train)} Zeilen")
        print(f"Validierungsdaten: {len(self.df_val)} Zeilen")
        print(f"Testdaten: {len(self.df_test)} Zeilen")
        print(f"Anzahl der Trainingsfenster: {self.n_windows}")
        print(f"Study Name: {self.study.study_name}")
        
        if isinstance(self.model, StatsForecast):
            first_model = self.model.models[0]
            print(f"StatsForecast Modell: {first_model.__class__.__name__} mit Order {first_model.order} und Seasonal Order {first_model.seasonal_order}")
            self.model_name = first_model.__class__.__name__

        elif isinstance(self.model, MLForecast):
            first_model = list(self.model.models.values())[0]
            print(f"MLForecast Modell: {first_model.steps[-1][1].__class__.__name__} mit Lags {self.lags}")
            self.model_name = first_model.steps[-1][1].__class__.__name__
            
        else:
            print(f"MLForecast Modell: {self.model.models[0].named_steps['model'].__class__.__name__} mit Lags {self.lags}")
        

    def _open_study_db(self, db_name):
        storage = f'sqlite:///{db_name}'
        study_name_list = optuna.study.get_all_study_names(storage=storage)
        print(study_name_list)

        study_name = next((name for name in study_name_list
                           if f'points{self.ocean_points}' in name and f'h{self.h}' in name), None)

        if study_name is None:
            raise ValueError(
                f"Keine passende Studie gefunden für points{self.ocean_points} und h{self.h} in {db_name}.\n"
                f"Verfügbare Studien: {study_name_list}"
            )
        return optuna.load_study(study_name=study_name, storage=storage)

    def _best_params(self, study):
        return study.best_trial.params

    def visualize_study(self):
        import optuna.visualization as vis
        vis.plot_optimization_history(self.study).show()
        vis.plot_param_importances(self.study).show()
        vis.plot_parallel_coordinate(
            self.study,
            params=[p for p, v in self.best_params.items() if isinstance(v, (int, float, str))]
        ).show()

    def _model(self):
        params = self.best_params.copy()

        if self.model_type == "sarima":
            # Use just the columns ['ds', 'unique_id', 'y'] for SARIMA
            
            return StatsForecast(
                models=[ARIMA(
                    order=(params["p"], params["d"], params["q"]),
                    seasonal_order=(params["P"], params["D"], params["Q"]),
                    season_length=params["s"]
                )],
                freq="h",
                n_jobs=self.n_jobs,
            )

        

        if self.model_type == "xgboost":
            model = XGBRegressor(**params, verbosity=0)
        elif self.model_type == "lgbm":
            model = LGBMRegressor(**params, verbosity=-1)
        elif self.model_type == "random_forest":
            model = RandomForestRegressor(**params)
        elif self.model_type == "svr":
            model = SVR(**params)
        elif self.model_type == "lr":
            degree = params.pop("degree")
            interaction_only = params.pop("interaction_only")
            include_bias = params.pop("include_bias")
            positive = params.pop("positive", False)
            poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
            model = Pipeline([
                #("poly", poly),
                ("scaler", StandardScaler()),
                ("model", LinearRegression(n_jobs=self.n_jobs)) #self.n_jobs 
            ])
            return MLForecast(models=[model], freq="h", lags=self.lags)
        else:
            raise ValueError("Unbekannter Modelltyp.")

        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])

        return MLForecast(models=[pipe], freq="h", lags=self.lags)

    # def fit(self, df=None):
    #     if isinstance(self.model, StatsForecast):
    #         self.model.fit(df=df)
    #     elif isinstance(self.model, MLForecast):
    #         self.model.fit(df=df, keep_last_n=self.h, static_features=[])
    #     elif df is None:
    #         raise ValueError("Für StatsForecast und MLForecast muss ein DataFrame übergeben werden.")
    #     else:
    #         raise ValueError("Unbekannter Modelltyp.")

    def create_forecast_df(self):
        

        all_forecasts = []
        for i in tqdm(range(0, len(self.df_val), self.h), desc="Rolling Window Progress"):
            if len(self.df_val) - i < self.h:
                continue

            history = pd.concat([self.df_train, self.df_val.iloc[:i]])

            if isinstance(self.model, StatsForecast):
                history = history[["ds", "unique_id", "y"]]
                self.model.fit(history)
                #exog_cols = [c for c in self.df_val.columns if c not in ["unique_id", "ds", "y"]]
                #X_exog = self.df_val.iloc[i:i + self.h][["unique_id", "ds"] + exog_cols]
                preds = self.model.predict(h=self.h)
                
            elif isinstance(self.model, MLForecast):
                self.model.fit(history, static_features=[])
                base_future = self.model.make_future_dataframe(h=self.h)
                exog_cols = [c for c in self.df_val.columns if c not in ["unique_id", "ds", "y"]]
                exog_future = self.df_val[["unique_id", "ds"] + exog_cols]
                future_df = base_future.merge(exog_future, on=["unique_id", "ds"], how="left").fillna(0) # dropna statt fillna(0) ?
                preds = self.model.predict(h=self.h, new_df=history, X_df=future_df)
            else:
                raise ValueError("Unbekannter Modelltyp.")

            fc_col = [c for c in preds.columns if c not in ["unique_id", "ds"]][0]
            preds = preds.rename(columns={fc_col: "forecast"})
            mask = (self.df_val["ds"] >= preds["ds"].min()) & (self.df_val["ds"] <= preds["ds"].max())
            true_block = self.df_val.loc[mask, ["unique_id", "ds", "y"]]
            merged = true_block.merge(preds[["unique_id", "ds", "forecast"]], on=["unique_id", "ds"])
            all_forecasts.append(merged)

        self.df_all = pd.concat(all_forecasts).sort_values("ds")
        return self.df_all




    def cross_validation_results(self):
        """
        Führt eine Cross-Validation mit Zeitmessung durch und gibt die Ergebnisse zurück.
        Für MLForecast und StatsForecast werden Fit- und Predict-Zeiten pro Fenster mitgeloggt.
        """
        

        df_cross = self.df_train
        df_cross = pd.concat([df_cross, self.df_val], ignore_index=True).dropna()

        if isinstance(self.model, StatsForecast):
            # Use just the columns ['ds', 'unique_id', 'y'] for SARIMA
            df_cross = df_cross[["ds", "unique_id", "y"]]
            all_results = []
            ##display(df_cross)
            for i in tqdm(range(self.n_windows), desc="StatsForecast CV mit Zeitmessung"):
                start_idx = i * self.step_size
                train_df = df_cross.iloc[:start_idx + self.horizon]
                test_df = df_cross.iloc[start_idx + self.horizon:start_idx + 2 * self.horizon]

                if len(test_df) < self.horizon:
                    continue

            

                t0 = time.time()
                self.model.fit(df=train_df)
                fit_time = time.time() - t0

                t1 = time.time()
                preds = self.model.predict(h=self.horizon, 
                                           #X_df=test_df.drop(columns=["y"])
                                           )
                predict_time = time.time() - t1

                preds["fit_time"] = fit_time
                preds["predict_time"] = predict_time
                preds["cutoff"] = df_cross["ds"].iloc[start_idx + self.horizon - 1]

                merged = test_df[["unique_id", "ds", "y"]].merge(preds, on=["unique_id", "ds"], how="left")
                # for col in ["fit_time", "predict_time", "cutoff"]:
                #     merged[col] = preds[col].iloc[0]

                all_results.append(merged)

            results = pd.concat(all_results).sort_values("ds").reset_index(drop=True)
            

            return results

        elif isinstance(self.model, MLForecast):
            all_results = []
            for i in tqdm(range(self.n_windows), desc="MLForecast CV mit Zeitmessung"):
                start_idx = i * self.step_size
                train = df_cross.iloc[:start_idx + self.horizon]
                test = df_cross.iloc[start_idx + self.horizon:start_idx + 2 * self.horizon]

                if len(test) < self.horizon:
                    continue

                t0 = time.time()
                self.model.fit(train, static_features=[])
                fit_time = time.time() - t0

                base_future = self.model.make_future_dataframe(h=self.horizon)
                exog_cols = [c for c in df_cross.columns if c not in ["unique_id", "ds", "y"]]
                X_df = test[["unique_id", "ds"] + exog_cols] if exog_cols else None

                t1 = time.time()
                preds = self.model.predict(
                    h=self.horizon,
                    new_df=train,
                    X_df=X_df
                )
                predict_time = time.time() - t1

                preds["fit_time"] = fit_time
                preds["predict_time"] = predict_time
                preds["cutoff"] = df_cross["ds"].iloc[start_idx + self.horizon - 1]

                # Merge mit Ground Truth (y)
                merged = test[["unique_id", "ds", "y"]].merge(preds, on=["unique_id", "ds"], how="left")
                all_results.append(merged)

            results = pd.concat(all_results).sort_values("ds").reset_index(drop=True)
            return results

        else:
            raise NotImplementedError("Unbekannter Modelltyp für Cross-Validation.")
        


    def get_model_statistics(self, results_df=None):

        """
        Gibt ein DataFrame mit den wichtigsten Modellstatistiken zurück.
        """
        if results_df is None or results_df.empty:
            #results_df = self.cross_validation_results()
            print("Warnung: Keine Ergebnisse für die Modellstatistik. Cross-Validation wird durchgeführt.")


        forecast_col = [c for c in results_df.columns if c not in ["unique_id", "ds", 'y', 'cutoff', 'fit_time', 'predict_time']][0]

        bins=[1, 1.25, 1.5, 2.0]

        y_true = results_df["y"].values
        y_pred = results_df[forecast_col].values

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

        # Anzahl Parameter
        num_params = len(self.best_params) if self.best_params is not None else np.nan
        model_name = [c for c in results_df.columns if c not in ["unique_id", "ds", 'y', 'cutoff', 'fit_time', 'predict_time']][0]
            
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
            "Num Params": num_params
        }

        return pd.DataFrame([stats])
        
    def plot_cross_validation_results(self, results=None, start_date=None, end_date=None, title="Forecast vs. Actual Water Level"):
        
        if results is None or results.empty:
            results = self.cross_validation_results()

        if start_date is not None:
            results = results[results['ds'] >= start_date]
        if end_date is not None:
            results = results[results['ds'] <= end_date]

        # Entferne unendliche Werte und ersetze durch NaN, dann drop
        results = results.replace([np.inf, -np.inf], np.nan)
        forecast_col = [c for c in results.columns if c not in ['unique_id', 'ds', 'cutoff', 'y']][0]
        # Nur Zeilen behalten, die sowohl y als auch Forecast finite haben
        valid_mask = results['y'].notna() & results[forecast_col].notna()
        if not valid_mask.any():
            print("Warnung: Nach dem Filtern sind keine gültigen y-/Forecast-Werte vorhanden. Plot wird übersprungen.")
            return
        results = results.loc[valid_mask].copy()

        # Jetzt erst plotten
        flood_levels = [
            (1.0, 1.25, 'yellow', 'storm surge'),
            (1.25, 1.5, 'orange', 'medium storm surge'),
            (1.5, 2.0, 'red', 'heavy storm surge'),
            (2.0, 3.5, 'darkred', 'very heavy storm surge'),
        ]

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(results['ds'], results[forecast_col], label='Forecasts', alpha=0.7, color=PRED_COLOR)
        ax.plot(results['ds'], results['y'], label='True Values', alpha=0.7, color=TRUE_COLOR)

        ax.fill_between(results['ds'], results['y'], results[forecast_col],
                        color='grey', alpha=0.3)

        for y0, y1, color, label in flood_levels:
            ax.axhspan(y0, y1, facecolor=color, alpha=0.2, label=label)

        for i in range(0, len(results), self.horizon):
            label = 'begin new forecast' if i == 0 else None
            ax.axvline(x=results['ds'].iloc[i], color='grey', linestyle='--', linewidth=1.5, label=label, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # sichere y-Limits berechnen (min/max über y und forecast)
        y_min = min(results['y'].min(), results[forecast_col].min())
        y_max = max(results['y'].max(), results[forecast_col].max())
        ax.set_ylim(y_min - 0.5, y_max + 0.5)

        ax.set_xlabel("Time")
        ax.set_ylabel("Water Level [m]")

        ax.set_title(title)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        fig.autofmt_xdate()
        ax.grid(True)
        plt.tight_layout()
        if start_date or end_date is None:
            start_date = results['ds'].min().strftime("%Y-%m-%d")
            end_date = results['ds'].max().strftime("%Y-%m-%d")
        plt.savefig(f"../thesis_plots/{self.trial_name}_ML_cross_validation_{self.model_name}_h{self.h}_startdate{start_date}_enddate{end_date}.png", dpi=300)
        #plt.show()



    def plot_test_forecast(self):
        
        
        df_train_val = pd.concat([self.df_train, self.df_val])
        
        

        if isinstance(self.model, StatsForecast):
            df_train_val.dropna(inplace=True)
            future_df = self.df_test.drop(columns=["y"])
            self.model.fit(df=df_train_val)
            preds = self.model.predict(h=self.h, X_df=future_df)
            fc_col = [c for c in preds.columns if c not in ["unique_id", "ds"]][0] # get model column e.g. SARIMA
            preds = preds.rename(columns={fc_col: "forecast"})
            results = self.df_test[["unique_id", "ds", "y"]].merge(preds, on=["unique_id", "ds"], how="left")

        elif isinstance(self.model, MLForecast):
            self.model.fit(df=df_train_val, static_features=[])
            base_future = self.model.make_future_dataframe(h=self.h)
            future_df = base_future.merge(self.df_test, on=["unique_id", "ds"], how="left").fillna(0)
            preds = self.model.predict(h=self.h, new_df=df_train_val, X_df=future_df)
            fc_col = [c for c in preds.columns if c not in ["unique_id", "ds"]][0] # get model column e.g. LinearRegression
            preds = preds.rename(columns={fc_col: "forecast"})
            results = self.df_test[["unique_id", "ds", "y"]].merge(preds, on=["unique_id", "ds"], how="left")

        # --- Fehlerabfang: Keine Daten oder nur NaN ---
        if results.empty or results['y'].dropna().empty or results['forecast'].dropna().empty:
            print("Warnung: Keine gültigen Werte für y oder forecast im Test-DataFrame. Plot wird übersprungen.")
            return
           
        # Highlight storm surge classes with colored bands
        flood_levels = [
            (1.0, 1.25, 'yellow', 'storm surge'),
            (1.25, 1.5, 'orange', 'medium storm surge'),
            (1.5, 2.0, 'red', 'heavy storm surge'),
            (2.0, 3.5, 'darkred', 'very heavy storm surge'),
        ]

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(results['ds'], results['forecast'], label='Forecast', alpha=0.7, color=PRED_COLOR)
        ax.plot(results['ds'], results['y'], label='True Values', color=TRUE_COLOR, alpha=0.7)
        ax.fill_between(results["ds"], results["y"], results["forecast"],
                         color='lightgrey', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        for y0, y1, color, label in flood_levels:
            ax.axhspan(y0, y1, facecolor=color, alpha=0.2, label=label)

        for i in range(0, len(results), self.h):
            label = 'begin new forecast' if i == 0 else None
            ax.axvline(x=results["ds"].iloc[i], color='gray', linestyle='--', linewidth=1.5, label=label)

        ax.set_xlabel("Time")
        ax.set_ylabel("Water Level [m]")
        ax.set_title("Forecast vs. Actual Water Level")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3)
        ax.set_ylim(results['y'].min() - 0.5, results['y'].max() + 0.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(10)) # make y-axis labels readable
        ax.xaxis.set_major_locator(plt.MaxNLocator(10)) # make x-axis labels readable
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        fig.autofmt_xdate()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"../thesis_plots/{self.trial_name}_ML_test_forecast_{self.model_name}_h{self.h}.png", dpi=300)
        #plt.show()


    def plot_rmse_per_cutoff(self, results=None):

        """
        Plottet den RMSE pro Cross-Validation-Fenster mit Mittelwert, Standardabweichung
        und Markierung von Sturmflut-Fenstern (y > 1.0).
        """
        if results is None or results.empty:
            results = self.cross_validation_results()

        forecast_col = [c for c in results.columns if c not in ['unique_id', 'ds', 'cutoff', 'y']][0]

        # RMSE pro Fenster berechnen
        grouped = results.groupby('cutoff')
        errors = grouped.apply(lambda x: mean_squared_error(x['y'], x[forecast_col]))

        mean_rmse = errors.mean()
        std_rmse = errors.std()

        # Identifiziere Sturmflut-Fenster (mind. 1 y > 1.0)
        sturmflut_cutoffs = grouped.filter(lambda x: (x['y'] >= 1.0).any())['cutoff'].unique()

        # Plot vorbereiten
        fig, ax = plt.subplots(figsize=(12, 5))
        errors.plot(marker='o', label='RMSE for each window', color='blue', ax=ax)

        ax.axhline(y=mean_rmse, color='green', linestyle='--', label=f'Ø RMSE = {mean_rmse:.3f}')
        ax.fill_between(errors.index, mean_rmse - std_rmse, mean_rmse + std_rmse,
                        color='green', alpha=0.2, label=f'±1 Std = {std_rmse:.3f}')

        # Farbig markieren: Sturmflut-Fenster
        for cutoff in sturmflut_cutoffs:
            if cutoff in errors.index:
                label = 'storm surge window' if cutoff == sturmflut_cutoffs[0] else None
                ax.axvspan(cutoff, cutoff + pd.Timedelta(hours=self.h), color='red', alpha=0.15, label=label)
                #plt.axvspan(cutoff, cutoff, color='red', alpha=0.15, linewidth=10)

        ax.set_title("RMSE per Cross-Validation Window")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Cutoff Time")
        ax.yaxis.set_major_locator(plt.MaxNLocator(10)) # make y-axis labels readable
        ax.xaxis.set_major_locator(plt.MaxNLocator(10)) # make x-axis labels readable
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        fig.autofmt_xdate()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"../thesis_plots/{self.trial_name}_ML_rmse_per_cutoff_{self.model_name}_h{self.h}.png", dpi=300)
        #plt.show()

        
    def plot_multiclass_recall_per_cutoff(self, results_df=None):

        """
        Plottet den macro-Recall pro Cross-Validation-Fenster mit mehrklassiger Klassifikation.
        
        Bins definieren die Klassengrenzen.
        """
        if results_df is None or results_df.empty:
            results_df = self.cross_validation_results()
            
        recall_per_cutoff = []
        grouped = results_df.groupby("cutoff")
        bins=[1, 1.25, 1.5, 2.0]
        forecast_col = [c for c in results_df.columns if c not in ['unique_id', 'ds', 'cutoff', 'y']][0]

        for cutoff, group in grouped:
            y_true = group["y"].values
            y_pred = group[forecast_col].values

            y_true_class = np.digitize(y_true, bins)
            y_pred_class = np.digitize(y_pred, bins)

            # Falls eine Klasse im y_true fehlt, wird Recall mit NaN vermieden durch try-except
            try:
                recall = recall_score(y_true_class, y_pred_class, average="macro", zero_division=0)
            except ValueError:
                recall = np.nan  # wenn keine Klassen vorhanden

            recall_per_cutoff.append((cutoff, recall))

        # In DataFrame umwandeln
        recall_df = pd.DataFrame(recall_per_cutoff, columns=["cutoff", "recall"])

        # Calculating average recall over all data
        y_true_all = results_df["y"].values
        y_pred_all = results_df[forecast_col].values
        y_true_class_all = np.digitize(y_true_all, bins)
        y_pred_class_all = np.digitize(y_pred_all, bins)
        recall = recall_score(y_true_class_all, y_pred_class_all, average="macro")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(recall_df["cutoff"], recall_df["recall"], marker="o", color="blue", label="Recall")
        ax.axhline(y=recall, color='gray', linestyle='--', label='Recall {:.3f}'.format(recall))

        # Farbig markieren: Sturmflut-Fenster
        sturmflut_cutoffs = results_df[results_df["y"] >= 1.0]["cutoff"].unique()
        for cutoff in sturmflut_cutoffs:
            if cutoff in recall_df["cutoff"].values:
                label = 'storm surge window' if cutoff == sturmflut_cutoffs[0] else None
                ax.axvspan(cutoff, cutoff + pd.Timedelta(hours=self.h), color='red', alpha=0.15, label=label)

        ax.set_title("Recall per window ")
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("Recall (macro)")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(plt.MaxNLocator(10)) # make y-axis labels readable
        ax.xaxis.set_major_locator(plt.MaxNLocator(10)) # make x-axis labels readable
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        fig.autofmt_xdate()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"../thesis_plots/{self.trial_name}_ML_multiclass_recall_per_cutoff_{self.model_name}_h{self.h}.png", dpi=300)
        #plt.show()

    def get_coefficients(self):
        if not isinstance(self.model, MLForecast):
            return None
            #raise ValueError("Nur für MLForecast-Modelle verfügbar.")
        
        

        pipe = self.model.models_[self.model_name]

        reg = pipe.named_steps['model']

        # Feature-Namen bestimmen
        try:
            poly = pipe.named_steps['poly']
            feature_names = poly.get_feature_names_out(self.model.ts.features_order_)
        except Exception:
            feature_names = getattr(self.model.ts, "features_order_", None)
            if feature_names is None:
                feature_names = [f"x{i}" for i in range(reg.n_features_in_)] if hasattr(reg, "n_features_in_") else []

        # Lineare Modelle
        if hasattr(reg, 'coef_'):
            return pd.DataFrame({
                'feature': feature_names,
                'coefficient': reg.coef_
            })

        # Baumbasierte Modelle: Feature Importances
        elif hasattr(reg, 'feature_importances_'):
            return pd.DataFrame({
                'feature': feature_names,
                'importance': reg.feature_importances_
            }).sort_values('importance', ascending=False)

        # XGBoost: Booster Feature Importances (falls vorhanden)
        elif hasattr(reg, 'get_booster'):
            booster = reg.get_booster()
            fmap = booster.get_score(importance_type='weight')
            importances = [fmap.get(f'f{i}', 0) for i in range(len(feature_names))]
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

        # SVR mit linearem Kernel
        elif hasattr(reg, 'coef_'):
            return pd.DataFrame({
                'feature': feature_names,
                'coefficient': reg.coef_.ravel()
            })

        else:
            raise ValueError("Dieses Modell unterstützt keine Feature-Koeffizienten oder Importances.")

    def plot_coefficients(self, coeff_df=None):
        """
        Plottet die Koeffizienten oder Feature-Importances des Modells.
        """

        if not isinstance(self.model, MLForecast):
            print("Nur für MLForecast-Modelle verfügbar.")
            return None
        
        if coeff_df is None:
            try:
                coeff_df = self.get_coefficients()
            except ValueError as e:
                print(str(e))
                return None

        if coeff_df.empty or coeff_df is None:
            print("Keine Koeffizienten oder Importances verfügbar.")
            return None

        n = 20
        # Plot n-largest coefficients or importances
        # Bestimme, welche Spalte verwendet werden soll
        value_col = 'coefficient' if 'coefficient' in coeff_df.columns else 'importance'

        # Plot n-largest
        fig, ax = plt.subplots(figsize=(12, len(coeff_df.nlargest(n, value_col)) * 0.3))
        coeff_df.nlargest(n, value_col).plot.barh(x='feature', y=value_col, ax=ax)
        ax.set_xlabel("Coefficient" if value_col == 'coefficient' else "Feature Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Largest Coefficients/Importances for {self.model_name}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"../thesis_plots/{self.trial_name}_ML_coefficients_{self.model_name}_h{self.h}_largest.png", dpi=300)
        #plt.show()

        plt.clf()
        plt.close(fig)

        # Plot n-smallest
        fig, ax = plt.subplots(figsize=(12, len(coeff_df.nsmallest(n, value_col)) * 0.3))
        coeff_df.nsmallest(n, value_col).plot.barh(x='feature', y=value_col, ax=ax)
        ax.set_xlabel("Coefficient" if value_col == 'coefficient' else "Feature Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Smallest Coefficients/Importances for {self.model_name}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"../thesis_plots/{self.trial_name}_ML_coefficients_{self.model_name}_h{self.h}_smallest.png", dpi=300)
        #plt.show()

    
    

        



# ## Linear Regression


folds = {
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
    

gap = pd.Timedelta(hours=168)


model_stats_all = pd.DataFrame()

model = EvaluationMLModel(
    model_type='lr',  # Change this to 'xgboost', 'lgbm', 'svr', 'lr', or 'sarima'
    df=df,
    Versuch="1.1",
    h=24,
    trial_name="Versuch1")

results = model.cross_validation_results()
model_stats = model.get_model_statistics(results)
#display(model_stats)
model.plot_cross_validation_results(results)
for surge_name, cutoff in folds.items():
    model.plot_cross_validation_results(results, start_date=cutoff - gap, end_date=cutoff + gap)
model.plot_test_forecast()
model.plot_rmse_per_cutoff(results)
model.plot_multiclass_recall_per_cutoff(results)
model.visualize_study()
model.plot_coefficients()

result_df = pd.DataFrame()
for surge_name, cutoff in folds.items():
    start_date = cutoff - gap
    end_date = cutoff + gap 

    results_surge = results[(results['ds'] >= start_date) & (results['ds'] <= end_date)]
    print(f"Results for {surge_name} from {start_date} to {end_date}:")
    model_surge_stats = model.get_model_statistics(results_surge)
    
    model_surge_stats['fold'] = surge_name
    result_df = pd.concat([result_df, model_surge_stats], ignore_index=True)
    model_name = model_surge_stats['Model'].iloc[0]
    
    

#display(result_df)
result_df_mean = result_df.mean(numeric_only=True).to_frame().T
result_df_mean['Model'] = model_name
#display(result_df_mean)
model_stats_all = pd.concat([model_stats_all, result_df_mean], ignore_index=True)
#display(model_stats_all)
# BARPLOT
# plot barplot
fig = plt.figure(figsize=(10, 6))
plt.bar(result_df['fold'], result_df['RMSE'], color=TRUE_COLOR, alpha=0.7)
rmse = result_df['RMSE'].mean()
std_value = result_df['RMSE'].std()

plt.axhspan(rmse - std_value, rmse + std_value, 
            color='green', alpha=0.2, 
            label=f'±1 STD: {std_value:.4f}')
plt.axhline(rmse, color='red', linestyle='--', label=f'Mean RMSE: {rmse:.3f}')
plt.legend()
plt.xlabel('Fold')
plt.ylabel('RMSE [m]')
plt.title('RMSE per Fold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"../thesis_plots/{model.trial_name}_ML_rmse_per_fold_{model.model_name}_h{model.h}.png", dpi=300)
#plt.show()










# ## Light Gradient Boosting Machine


model = EvaluationMLModel(
    model_type='lgbm',  # Change this to 'xgboost', 'lgbm', 'svr', 'lr', or 'sarima'
    df=df,
    Versuch="1.1",
    h=24,
    trial_name="Versuch1")

results = model.cross_validation_results()
model_stats = model.get_model_statistics(results)
##display(model_stats)
model.plot_cross_validation_results(results)
for surge_name, cutoff in folds.items():
    model.plot_cross_validation_results(results, start_date=cutoff - gap, end_date=cutoff + gap, title=f"{surge_name}: Forecast vs. Actual Water Level")
model.plot_test_forecast()
model.plot_rmse_per_cutoff(results)
model.plot_multiclass_recall_per_cutoff(results)
model.visualize_study()
model.plot_coefficients()


result_df = pd.DataFrame()
for surge_name, cutoff in folds.items():
    start_date = cutoff - gap
    end_date = cutoff + gap 

    results_surge = results[(results['ds'] >= start_date) & (results['ds'] <= end_date)]
    print(f"Results for {surge_name} from {start_date} to {end_date}:")
    model_surge_stats = model.get_model_statistics(results_surge)
    
    model_surge_stats['fold'] = surge_name
    result_df = pd.concat([result_df, model_surge_stats], ignore_index=True)
    model_name = model_surge_stats['Model'].iloc[0]
    
    


##display(result_df)
result_df_mean = result_df.mean(numeric_only=True).to_frame().T
result_df_mean['Model'] = model_name
##display(result_df_mean)
model_stats_all = pd.concat([model_stats_all, result_df_mean], ignore_index=True)
##display(model_stats_all)
# BARPLOT
# plot barplot
fig = plt.figure(figsize=(10, 6))
plt.bar(result_df['fold'], result_df['RMSE'], color=TRUE_COLOR, alpha=0.7)
rmse = result_df['RMSE'].mean()
std_value = result_df['RMSE'].std()

plt.axhspan(rmse - std_value, rmse + std_value, 
            color='green', alpha=0.2, 
            label=f'±1 STD: {std_value:.4f}')
plt.axhline(rmse, color='red', linestyle='--', label=f'Mean RMSE: {rmse:.3f}')
plt.legend()
plt.xlabel('Fold')
plt.ylabel('RMSE [m]')
plt.title('RMSE per Fold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"../thesis_plots/{model.trial_name}_ML_rmse_per_fold_{model.model_name}_h{model.h}.png", dpi=300)
#plt.show()


# ## XGBoost


model = EvaluationMLModel(
    model_type='xgboost',  # Change this to 'xgboost', 'lgbm', 'svr', 'lr', or 'sarima'
    df=df,
    Versuch="1.1",
    h=24,
    trial_name="Versuch1")

results = model.cross_validation_results()
model_stats = model.get_model_statistics(results)
##display(model_stats)
model.plot_cross_validation_results(results)
for surge_name, cutoff in folds.items():
    model.plot_cross_validation_results(results, start_date=cutoff - gap, end_date=cutoff + gap)
model.plot_test_forecast()
model.plot_rmse_per_cutoff(results)
model.plot_multiclass_recall_per_cutoff(results)
model.visualize_study()
model.plot_coefficients()

result_df = pd.DataFrame()
for surge_name, cutoff in folds.items():
    start_date = cutoff - gap
    end_date = cutoff + gap 

    results_surge = results[(results['ds'] >= start_date) & (results['ds'] <= end_date)]
    print(f"Results for {surge_name} from {start_date} to {end_date}:")
    model_surge_stats = model.get_model_statistics(results_surge)
    
    model_surge_stats['fold'] = surge_name
    result_df = pd.concat([result_df, model_surge_stats], ignore_index=True)
    model_name = model_surge_stats['Model'].iloc[0]
    
    

##display(result_df)
result_df_mean = result_df.mean(numeric_only=True).to_frame().T
result_df_mean['Model'] = model_name
##display(result_df_mean)
model_stats_all = pd.concat([model_stats_all, result_df_mean], ignore_index=True)
##display(model_stats_all)
# BARPLOT
# plot barplot
fig = plt.figure(figsize=(10, 6))
plt.bar(result_df['fold'], result_df['RMSE'], color=TRUE_COLOR, alpha=0.7)
rmse = result_df['RMSE'].mean()
std_value = result_df['RMSE'].std()

plt.axhspan(rmse - std_value, rmse + std_value, 
            color='green', alpha=0.2, 
            label=f'±1 STD: {std_value:.4f}')
plt.axhline(rmse, color='red', linestyle='--', label=f'Mean RMSE: {rmse:.3f}')
plt.legend()
plt.xlabel('Fold')
plt.ylabel('RMSE [m]')
plt.title('RMSE per Fold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"../thesis_plots/{model.trial_name}_ML_rmse_per_fold_{model.model_name}_h{model.h}.png", dpi=300)
#plt.show()


# ## Random Rorest


model = EvaluationMLModel(
    model_type='random_forest',  # Change this to 'xgboost', 'lgbm', 'svr', 'lr', or 'sarima', random_forest
    df=df,
    Versuch="1.1",
    h=24,
    trial_name="Versuch1")

results = model.cross_validation_results()
model_stats = model.get_model_statistics(results)
##display(model_stats)
model.plot_cross_validation_results(results)
for surge_name, cutoff in folds.items():
    model.plot_cross_validation_results(results, start_date=cutoff - gap, end_date=cutoff + gap)
model.plot_test_forecast()
model.plot_rmse_per_cutoff(results)
model.plot_multiclass_recall_per_cutoff(results)
model.visualize_study()
model.plot_coefficients()

result_df = pd.DataFrame()
for surge_name, cutoff in folds.items():
    start_date = cutoff - gap
    end_date = cutoff + gap 

    results_surge = results[(results['ds'] >= start_date) & (results['ds'] <= end_date)]
    print(f"Results for {surge_name} from {start_date} to {end_date}:")
    model_surge_stats = model.get_model_statistics(results_surge)
    
    model_surge_stats['fold'] = surge_name
    result_df = pd.concat([result_df, model_surge_stats], ignore_index=True)
    model_name = model_surge_stats['Model'].iloc[0]
    
    

##display(result_df)
result_df_mean = result_df.mean(numeric_only=True).to_frame().T
result_df_mean['Model'] = model_name
##display(result_df_mean)
model_stats_all = pd.concat([model_stats_all, result_df_mean], ignore_index=True)
##display(model_stats_all)
# BARPLOT
# plot barplot
fig = plt.figure(figsize=(10, 6))
plt.bar(result_df['fold'], result_df['RMSE'], color=TRUE_COLOR, alpha=0.7)
rmse = result_df['RMSE'].mean()
std_value = result_df['RMSE'].std()

plt.axhspan(rmse - std_value, rmse + std_value, 
            color='green', alpha=0.2, 
            label=f'±1 STD: {std_value:.4f}')
plt.axhline(rmse, color='red', linestyle='--', label=f'Mean RMSE: {rmse:.3f}')
plt.legend()
plt.xlabel('Fold')
plt.ylabel('RMSE [m]')
plt.title('RMSE per Fold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"../thesis_plots/{model.trial_name}_ML_rmse_per_fold_{model.model_name}_h{model.h}.png", dpi=300)
#plt.show()


# ## Sarima


model = EvaluationMLModel(
    model_type='sarima',  # Change this to 'xgboost', 'lgbm', 'svr', 'lr', or 'sarima', random_forest
    df=df,
    Versuch="1.1",
    h=24,
    trial_name="Versuch1")

results = model.cross_validation_results()
model_stats = model.get_model_statistics(results)
##display(model_stats)
model.plot_cross_validation_results(results)
for surge_name, cutoff in folds.items():
    model.plot_cross_validation_results(results, start_date=cutoff - gap, end_date=cutoff + gap)
model.plot_test_forecast()
model.plot_rmse_per_cutoff(results)
model.plot_multiclass_recall_per_cutoff(results)
model.visualize_study()
model.plot_coefficients()

result_df = pd.DataFrame()
for surge_name, cutoff in folds.items():
    start_date = cutoff - gap
    end_date = cutoff + gap 

    results_surge = results[(results['ds'] >= start_date) & (results['ds'] <= end_date)]
    print(f"Results for {surge_name} from {start_date} to {end_date}:")
    model_surge_stats = model.get_model_statistics(results_surge)
    
    model_surge_stats['fold'] = surge_name
    result_df = pd.concat([result_df, model_surge_stats], ignore_index=True)
    model_name = model_surge_stats['Model'].iloc[0]
    
    

##display(result_df)
result_df_mean = result_df.mean(numeric_only=True).to_frame().T
result_df_mean['Model'] = model_name
##display(result_df_mean)
model_stats_all = pd.concat([model_stats_all, result_df_mean], ignore_index=True)
##display(model_stats_all)
# BARPLOT
# plot barplot
fig = plt.figure(figsize=(10, 6))
plt.bar(result_df['fold'], result_df['RMSE'], color=TRUE_COLOR, alpha=0.7)
rmse = result_df['RMSE'].mean()
std_value = result_df['RMSE'].std()

plt.axhspan(rmse - std_value, rmse + std_value, 
            color='green', alpha=0.2, 
            label=f'±1 STD: {std_value:.4f}')
plt.axhline(rmse, color='red', linestyle='--', label=f'Mean RMSE: {rmse:.3f}')
plt.legend()
plt.xlabel('Fold')
plt.ylabel('RMSE [m]')
plt.title('RMSE per Fold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"../thesis_plots/{model.trial_name}_ML_rmse_per_fold_{model.model_name}_h{model.h}.png", dpi=300)
#plt.show()


# ## Support Vector Regression


model = EvaluationMLModel(
    model_type='svr',  # Change this to 'xgboost', 'lgbm', 'svr', 'lr', or 'sarima', random_forest
    df=df,
    Versuch="1.1",
    h=24,
    trial_name="Versuch1")

results = model.cross_validation_results()
model_stats = model.get_model_statistics(results)
##display(model_stats)
model.plot_cross_validation_results(results)
for surge_name, cutoff in folds.items():
    model.plot_cross_validation_results(results, start_date=cutoff - gap, end_date=cutoff + gap)
model.plot_test_forecast()
model.plot_rmse_per_cutoff(results)
model.plot_multiclass_recall_per_cutoff(results)
model.visualize_study()
model.plot_coefficients()

result_df = pd.DataFrame()
for surge_name, cutoff in folds.items():
    start_date = cutoff - gap
    end_date = cutoff + gap 

    results_surge = results[(results['ds'] >= start_date) & (results['ds'] <= end_date)]
    print(f"Results for {surge_name} from {start_date} to {end_date}:")
    model_surge_stats = model.get_model_statistics(results_surge)
    
    model_surge_stats['fold'] = surge_name
    result_df = pd.concat([result_df, model_surge_stats], ignore_index=True)
    model_name = model_surge_stats['Model'].iloc[0]
    
    


##display(result_df)
result_df_mean = result_df.mean(numeric_only=True).to_frame().T
result_df_mean['Model'] = model_name
##display(result_df_mean)
model_stats_all = pd.concat([model_stats_all, result_df_mean], ignore_index=True)
##display(model_stats_all)
# BARPLOT
# plot barplot
fig = plt.figure(figsize=(10, 6))
plt.bar(result_df['fold'], result_df['RMSE'], color=TRUE_COLOR, alpha=0.7)
rmse = result_df['RMSE'].mean()
std_value = result_df['RMSE'].std()

plt.axhspan(rmse - std_value, rmse + std_value, 
            color='green', alpha=0.2, 
            label=f'±1 STD: {std_value:.4f}')
plt.axhline(rmse, color='red', linestyle='--', label=f'Mean RMSE: {rmse:.3f}')
plt.legend()
plt.xlabel('Fold')
plt.ylabel('RMSE [m]')
plt.title('RMSE per Fold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"../thesis_plots/{model.trial_name}_ML_rmse_per_fold_{model.model_name}_h{model.h}.png", dpi=300)
#plt.show()


##display(model_stats_all)


##display(model_stats_all.sort_values(by='MAE', ascending=True))
##display(model_stats_all.sort_values(by='Recall', ascending=False))
model_stats_all.to_csv("../thesis_plots/Versuch1_ML_model_statistics_all_models.csv", index=False)


