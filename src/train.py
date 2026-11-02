import os
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from src.data import load_clean_data
from src.features import (
    strategy_drop, strategy_ordinal, strategy_ohe, strategy_ohe_poly,
    scale_train_test,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def train_all():
    df = load_clean_data()
    print(f"Данные после очистки: {df.shape[0]} строк, {df.shape[1]} колонок")

    results = {}
    artifacts = {}

    strategies = {
        "Linear + Drop": strategy_drop,
        "Linear + Ordinal": strategy_ordinal,
        "Linear + OHE": strategy_ohe,
    }

    for name, fn in strategies.items():
        print(f"\n>>> {name}")
        X, y = fn(df)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_tr_s, X_te_s, scaler = scale_train_test(X_tr, X_te)

        model = LinearRegression()
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        results[name] = evaluate(y_te, y_pred)
        artifacts[name] = {
            "model": model,
            "scaler": scaler,
            "features": list(X.columns),
        }
        print(f"  R²={results[name]['r2']:.4f}  RMSE={results[name]['rmse']:.2f}  MAPE={results[name]['mape']:.2f}%")

    print("\n>>> Linear + OHE + Polynomial")
    X, y, feat_names = strategy_ohe_poly(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr_s, X_te_s, scaler = scale_train_test(X_tr, X_te)

    model = LinearRegression()
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)

    results["Linear + OHE + Poly"] = evaluate(y_te, y_pred)
    artifacts["Linear + OHE + Poly"] = {
        "model": model,
        "scaler": scaler,
        "features": feat_names,
        "y_test": y_te.values,
        "y_pred": y_pred,
    }
    print(f"  R²={results['Linear + OHE + Poly']['r2']:.4f}  "
          f"RMSE={results['Linear + OHE + Poly']['rmse']:.2f}  "
          f"MAPE={results['Linear + OHE + Poly']['mape']:.2f}%")

    print("\n>>> RandomForest + OHE")
    X, y = strategy_ohe(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)

    results["RandomForest + OHE"] = evaluate(y_te, y_pred_rf)
    artifacts["RandomForest + OHE"] = {
        "model": rf,
        "features": list(X.columns),
        "y_test": y_te.values,
        "y_pred": y_pred_rf,
    }
    print(f"  R²={results['RandomForest + OHE']['r2']:.4f}  "
          f"RMSE={results['RandomForest + OHE']['rmse']:.2f}  "
          f"MAPE={results['RandomForest + OHE']['mape']:.2f}%")


    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for name, art in artifacts.items():
        fname = name.replace(" + ", "_").replace(" ", "_") + ".pkl"
        with open(os.path.join(MODELS_DIR, fname), "wb") as f:
            pickle.dump(art, f)

    print(f"\nМетрики сохранены в models/metrics.json")
    print(f"Модели сохранены в models/")


if __name__ == "__main__":
    train_all()