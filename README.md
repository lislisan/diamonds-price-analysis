# 💎 Анализ цен на бриллианты

Проект по анализу датасета [Diamonds](https://www.kaggle.com/datasets/shivam2503/diamonds) (54 000 бриллиантов):
EDA, сравнение стратегий кодирования категориальных признаков и интерактивный дашборд на Streamlit.

## Что внутри

- **4 стратегии** работы с категориальными (`cut`, `color`, `clarity`):
  1. Удаление
  2. OrdinalEncoder
  3. OneHotEncoding
  4. OHE + полином 2-й степени для `carat`, `x`, `y`, `z`
- **RandomForest** на OHE — как бенчмарк нелинейной модели
- **Streamlit-дашборд** с EDA, сравнением метрик и анализом ошибок


## Запуск локально (pip)

```bash
# 1. Клонировать репозиторий
git clone <url>
cd diamonds-price-analysis

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Скачать датасет
python -m src.data
# (потребуются Kaggle API credentials: kaggle.json в ~/.kaggle/)
# Альтернатива: скачать diamonds.csv с Kaggle вручную и положить в data/

# 4. Обучить модели
python -m src.train

# 5. Запустить дашборд
streamlit run app.py
```

Открой http://localhost:8501

Запуск в Docker
```bash
docker build -t diamonds-app .
docker run -p 8501:8501 diamonds-app
```