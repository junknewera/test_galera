import argparse
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from scripts.utils import merge_frames, feature_engineering, final_columns
import joblib


def load_model(model_path: str) -> TabNetClassifier:
    model = TabNetClassifier()
    model.load_model(model_path)
    return model


def infer(
    pq_path: str,
    context_path: str,
    model_path: str,
    output_path: str = "predictions.csv",
):
    # Загрузка данных
    pq = pd.read_parquet(pq_path)
    context = pd.read_csv(context_path)

    # Мерджинг и фичи
    merged = merge_frames(pq, context)
    df_fe = feature_engineering(merged)

    # Прогноз
    X = df_fe[final_columns].values
    model = load_model(model_path)
    preds = model.predict(X)

    # Сохраняем результат
    encoder = joblib.load("models/label_encoder.pkl")
    preds = encoder.inverse_transform(preds)
    df_fe["predicted_cus_class"] = preds
    df_fe[["date", "predicted_cus_class"]].to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on new data with TabNet model."
    )
    parser.add_argument("--pq_path", required=True, help="Path to test_task.parquet")
    parser.add_argument("--context_path", required=True, help="Path to context_df.csv")
    parser.add_argument(
        "--model_path", required=True, help="Path to trained TabNet model (zip)"
    )
    parser.add_argument(
        "--output_path", default="predictions.csv", help="Where to save predictions"
    )

    args = parser.parse_args()

    infer(
        pq_path=args.pq_path,
        context_path=args.context_path,
        model_path=args.model_path,
        output_path=args.output_path,
    )
