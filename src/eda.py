import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_price_dist(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["price"], kde=True, bins=50, color="steelblue", ax=ax)
    ax.set_title("Распределение цены")
    return fig


def plot_carat_vs_price(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="carat", y="price", alpha=0.3, color="teal", ax=ax)
    ax.set_title("Цена vs вес (carat)")
    return fig


def plot_cut_price(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="cut", y="price", palette="Set2", hue="cut", legend=False, ax=ax)
    ax.set_title("Цена по огранке (cut)")
    return fig


def plot_color_price(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="color", y="price", palette="Set2", hue="cut", legend=False, ax=ax)
    ax.set_title("Цена по цвету (color)")
    return fig


def plot_heatmap_cut_color(df):
    pivot = df.pivot_table(index="cut", columns="color", values="price", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title("Средняя цена: cut × color")
    return fig


def plot_corr(df):
    cols = ["carat", "depth", "table", "price", "x", "y", "z"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Матрица корреляций")
    return fig