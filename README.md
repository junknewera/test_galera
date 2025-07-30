# Class prediction

## Выгрузки

- **test_task.parquet** — Исторические значения показателя «класс продукта» cus_class в покупках.
- **context_df.csv** — Макропараметры экономики, с разбивкой по кварталам.

## Быстрый старт

1. **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Обучение модели** выполняется в ноутбуке [`notebooks/notebook_v3.ipynb`](notebooks/notebook_v3.ipynb).  
    Ноутбук загружает данные через функции из [`scripts/utils.py`](scripts/utils.py), конструирует признаки и обучает `TabNetClassifier`.

3. **Сохранение модели:**
    ```python
    model.save_model("models/tabnet_model.zip")
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    ```

4. **Инференс на новых данных** с помощью скрипта [`scripts/inference.py`](scripts/inference.py):
    ```bash
    python scripts/inference.py \
         --pq_path path/to/test_task.parquet \
         --context_path path/to/context_df.csv \
         --model_path models/tabnet_model.zip \
         --output_path predictions.csv
    ```
    Скрипт объединит выгрузки, построит признаки, прогонит их через модель, затем декодирует предсказания в исходные значения `cus_class` с помощью сохранённого `LabelEncoder` и сохранит итоговый CSV.

## Структура проекта

- `notebooks/` — Jupyter‑ноутбуки для обучения и экспериментов.
- `scripts/utils.py` — функции для загрузки, слияния и подготовки данных.
- `scripts/inference.py` — модуль для инференса.
- `models/` — папка для сохранения обученной модели и энкодера (создаётся во время обучения).