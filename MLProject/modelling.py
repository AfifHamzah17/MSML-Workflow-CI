import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import mlflow

def load_data(path):
    return pd.read_csv(path)

def train_and_log(data_path, n_estimators, max_depth):
    df = load_data(data_path)
    X = df.drop(columns=['mpg', 'name', 'origin', 'brand'], errors='ignore')
    y = df['mpg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Konfigurasi tracking lokal
    os.makedirs('mlruns', exist_ok=True)
    mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
    mlflow.set_experiment("auto_mpg_ci")

    # Mulai run baru dengan nested=True untuk hindari reuse run ID
    with mlflow.start_run(nested=True):
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Logging manual
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })
        mlflow.log_metric("test_r2", test_score)

        print(f"Test R²: {test_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    train_and_log(args.data_path, args.n_estimators, args.max_depth)