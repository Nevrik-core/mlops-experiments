import os
import shutil
from pathlib import Path

import joblib
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def push_metrics(pushgateway_url: str, run_id: str, acc: float, loss: float) -> None:
    registry = CollectorRegistry()
    g_acc = Gauge("mlflow_accuracy", "Model accuracy from MLflow runs", registry=registry)
    g_loss = Gauge("mlflow_loss", "Model loss (log_loss) from MLflow runs", registry=registry)

    g_acc.set(acc)
    g_loss.set(loss)

    push_to_gateway(
        pushgateway_url,
        job="mlflow_experiment",
        registry=registry,
        grouping_key={"run_id": run_id},
    )


def get_or_create_experiment(client: MlflowClient, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return client.create_experiment(name)


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    pushgateway_url = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_name = "iris-sgd"
    experiment_id = get_or_create_experiment(client, experiment_name)

    print("Tracking URI:", tracking_uri)
    print("Experiment ID:", experiment_id)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    learning_rates = [0.001, 0.01, 0.1]
    epochs_list = [50, 200, 800]

    runs = []

    for lr in learning_rates:
        for epochs in epochs_list:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                print(f"Started run_id={run_id} lr={lr} epochs={epochs}")

                clf = SGDClassifier(
                    loss="log_loss",
                    learning_rate="constant",
                    eta0=lr,
                    max_iter=epochs,
                    tol=None,
                    random_state=42,
                )
                clf.fit(X_train, y_train)

                proba = clf.predict_proba(X_test)
                preds = np.argmax(proba, axis=1)

                acc = float(accuracy_score(y_test, preds))
                loss = float(log_loss(y_test, proba))

                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("epochs", epochs)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("loss", loss)

                model_dir = Path("tmp_model")
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                ensure_dir(model_dir)

                model_path = model_dir / "model.joblib"
                joblib.dump(clf, model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model")

                push_metrics(pushgateway_url, run_id, acc, loss)

                print(f"Logged run_id={run_id} acc={acc:.4f} loss={loss:.4f}")
                runs.append((run_id, acc))

    best_run_id, best_acc = sorted(runs, key=lambda x: x[1], reverse=True)[0]
    print(f"Best run: {best_run_id} accuracy={best_acc:.4f}")

    best_dir = Path("../best_model").resolve()
    ensure_dir(best_dir)

    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=best_run_id,
        artifact_path="model",
    )

    downloaded_path = Path(downloaded_path)
    src = downloaded_path / "model.joblib"

    if not src.exists():
        candidates = list(downloaded_path.rglob("model.joblib"))
        if not candidates:
            raise FileNotFoundError(f"model.joblib not found in {downloaded_path}")
        src = candidates[0]

    dst = best_dir / "model.joblib"
    shutil.copy2(src, dst)
    print(f"Copied best model to: {dst}")


if __name__ == "__main__":
    main()