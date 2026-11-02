import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, StandardScaler
import numpy as np

CATEGORICAL = ["cut", "color", "clarity"]
NUMERIC = ["carat", "depth", "table", "x", "y", "z"]
POLY_COLS = ["carat", "x", "y", "z"]
LINEAR_COLS = ["depth", "table"]

CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


def strategy_drop(df):
    X = df.drop(columns=CATEGORICAL + ["price"])
    y = df["price"]
    return X, y


def strategy_ordinal(df):
    df_ = df.copy()
    encoder = OrdinalEncoder(categories=[CUT_ORDER, COLOR_ORDER, CLARITY_ORDER])
    df_[CATEGORICAL] = encoder.fit_transform(df_[CATEGORICAL])
    X = df_.drop(columns=["price"])
    y = df_["price"]
    return X, y


def strategy_ohe(df):
    df_ = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)
    X = df_.drop(columns=["price"])
    y = df_["price"]
    return X, y


def strategy_ohe_poly(df):
    df_ = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)
    y = df_["price"]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(df_[POLY_COLS])
    poly_names = poly.get_feature_names_out(POLY_COLS)

    X_lin = df_[LINEAR_COLS].values
    cat_cols = [c for c in df_.columns if c not in POLY_COLS + LINEAR_COLS + ["price"]]
    X_cat = df_[cat_cols].values

    X = np.hstack([X_poly, X_lin, X_cat])
    feature_names = list(poly_names) + LINEAR_COLS + cat_cols
    return X, y, feature_names


def scale_train_test(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler