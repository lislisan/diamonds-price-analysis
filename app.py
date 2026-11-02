"""Дашборд анализа diamonds."""
import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.data import load_clean_data, DATA_PATH
from src import eda

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

st.set_page_config(page_title="Diamonds Analysis", layout="wide")
st.title("💎 Анализ цен на бриллианты")


@st.cache_data
def get_data():
    if not os.path.exists(DATA_PATH):
        return None
    return load_clean_data()


@st.cache_data
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


df = get_data()
metrics = get_metrics()

if df is None:
    st.error(
        "Датасет не найден. Положи `diamonds.csv` в `data/` "
        "или запусти `python -m src.data`."
    )
    st.stop()

tab_eda, tab_metrics, tab_errors = st.tabs(["📊 EDA", "📈 Сравнение моделей", "🔍 Анализ ошибок"])


with tab_eda:
    st.header("Разведочный анализ")
    st.write(f"Строк после очистки: **{len(df):,}**")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(eda.plot_price_dist(df))
        st.pyplot(eda.plot_cut_price(df))
    with col2:
        st.pyplot(eda.plot_carat_vs_price(df))
        st.pyplot(eda.plot_color_price(df))

    st.pyplot(eda.plot_heatmap_cut_color(df))
    st.pyplot(eda.plot_corr(df))


with tab_metrics:
    st.header("Сравнение моделей")
    if metrics is None:
        st.warning(
            "Метрики не найдены. Запусти обучение:\n\n"
            "```bash\npython -m src.train\n```"
        )
    else:
        rows = [{"Модель": k, **v} for k, v in metrics.items()]
        df_m = pd.DataFrame(rows).set_index("Модель")
        st.dataframe(
            df_m.style.format({
                "r2": "{:.4f}", "rmse": "{:.2f}",
                "mae": "{:.2f}", "mape": "{:.2f}%",
            }),
            use_container_width=True,
        )

        st.subheader("R² по моделям")
        fig, ax = plt.subplots(figsize=(9, 4))
        df_m["r2"].plot(kind="bar", color="steelblue", ax=ax)
        ax.set_ylabel("R²")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)

        st.subheader("RMSE (доллары) — меньше лучше")
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        df_m["rmse"].plot(kind="bar", color="indianred", ax=ax2)
        ax2.set_ylabel("RMSE")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig2)


with tab_errors:
    st.header("Анализ ошибок лучшей модели")
    if metrics is None:
        st.warning("Сначала запусти `python -m src.train`.")
    else:
        best_name = max(metrics, key=lambda k: metrics[k]["r2"])
        st.write(f"Лучшая модель: **{best_name}** (R² = {metrics[best_name]['r2']:.4f})")

        fname = best_name.replace(" + ", "_").replace(" ", "_") + ".pkl"
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            st.warning("Артефакт лучшей модели не найден.")
        else:
            with open(path, "rb") as f:
                art = pickle.load(f)
            if "y_test" not in art:
                st.info("Для этой модели не сохранены предсказания на тесте.")
            else:
                y_t = art["y_test"]
                y_p = art["y_pred"]

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(y_t, y_p, alpha=0.3, s=10, color="teal")
                lim = [0, max(y_t.max(), y_p.max()) * 1.05]
                ax.plot(lim, lim, "r--", label="Идеал")
                ax.set_xlabel("Реальная цена")
                ax.set_ylabel("Предсказанная цена")
                ax.set_title("y_true vs y_pred")
                ax.legend()
                st.pyplot(fig)

                residuals = y_t - y_p
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                ax2.hist(residuals, bins=60, color="steelblue")
                ax2.set_xlabel("Остаток (true − pred)")
                ax2.set_ylabel("Частота")
                ax2.set_title("Распределение остатков")
                st.pyplot(fig2)

                st.write(f"Средний остаток: **{residuals.mean():.2f}**")
                st.write(f"Std остатков: **{residuals.std():.2f}**")