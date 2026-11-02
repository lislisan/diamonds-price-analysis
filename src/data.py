"""Загрузка и очистка датасета diamonds."""
import os
import pandas as pd

DATA_URL = "https://www.kaggle.com/datasets/shivam2503/diamonds"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diamonds.csv")
TARGET = "price"
CATEGORICAL = ["cut", "color", "clarity"]
NUMERIC = ["carat", "depth", "table", "x", "y", "z"]


def download_data():
    import opendatasets as od
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    od.download(DATA_URL, data_dir=os.path.dirname(DATA_PATH))
    # opendatasets сохраняет в data/diamonds/diamonds.csv — переносим
    src = os.path.join(os.path.dirname(DATA_PATH), "diamonds", "diamonds.csv")
    if os.path.exists(src):
        os.replace(src, DATA_PATH)
        os.rmdir(os.path.dirname(src))


def load_clean_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Датасет не найден: {DATA_PATH}\n"
            "Запусти `python -m src.data` для скачивания или положи diamonds.csv в data/."
        )
    df = pd.read_csv(DATA_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)].copy()

    df = df.drop_duplicates().reset_index(drop=True)

    return df


if __name__ == "__main__":
    download_data()
    print("Готово. Датасет в data/diamonds.csv")