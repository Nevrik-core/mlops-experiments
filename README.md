# HW9 — MLOps з ArgoCD, MLflow, MinIO, Postgres і PushGateway

## Опис

У цьому проєкті реалізовано MLOps-пайплайн у Kubernetes з використанням ArgoCD.

Було налаштовано:

- **Postgres** — backend store для MLflow
- **MinIO** — S3-сумісне сховище артефактів
- **MLflow** — трекінг експериментів, параметрів, метрик і моделей
- **PushGateway** — прийом метрик із Python-скрипта
- **ArgoCD** — GitOps-деплой застосунків у кластер

## Структура

Основні маніфести розміщені в:

- `argocd/applications/`
- `argocd/manifests/mlflow/`
- `argocd/manifests/namespaces/`
- `experiments/train_and_push.py`

## Що реалізовано

### 1. Деплой через ArgoCD

Через ArgoCD були розгорнуті:

- `postgres`
- `minio`
- `mlflow`
- `pushgateway`
- `namespaces`

### 2. MLflow

MLflow налаштований для роботи з:

- **Postgres** як backend store
- **MinIO** як artifact storage

MLflow зберігає:

- експерименти
- параметри запусків
- метрики
- артефакти моделі

### 3. Python-скрипт

Скрипт `train_and_push.py`:

- завантажує датасет Iris
- тренує кілька моделей `SGDClassifier`
- логить параметри та метрики в MLflow
- зберігає артефакт моделі
- пушить метрики в PushGateway
- знаходить найкращий запуск
- копіює найкращу модель у `best_model/model.joblib`

## Залежності

Файл `experiments/requirements.txt` містить основні бібліотеки:

- `mlflow==3.7.0`
- `scikit-learn`
- `pandas`
- `numpy`
- `joblib`
- `prometheus-client`
- `requests`

Додатково для роботи з MinIO потрібно було встановити:

```bash
pip install boto3
```

## Підготовка середовища

```bash
cd experiments
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install boto3
```

## Змінні середовища

Перед запуском скрипта використовувались такі змінні:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export PUSHGATEWAY_URL=http://localhost:9091
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadminpass
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```

## Port-forward

### MLflow

```bash
kubectl -n application port-forward svc/mlflow 5000:5000
```

### MinIO

```bash
kubectl -n application port-forward svc/minio 9000:9000
```

### PushGateway

```bash
kubectl -n monitoring port-forward svc/pushgateway 9091:9091
```

## Запуск навчання

```bash
cd experiments
source .venv/bin/activate
python train_and_push.py
```

## Результат

У результаті було успішно реалізовано:

- GitOps-деплой інфраструктури через ArgoCD
- трекінг експериментів у MLflow
- збереження артефактів у MinIO
- використання Postgres як backend store
- пуш метрик у PushGateway
- автоматичне збереження найкращої моделі локально

Після успішного запуску найкраща модель зберігається в:

```text
best_model/model.joblib
```

## Приклади перевірки

```bash
kubectl get applications.argoproj.io -A
kubectl get pods -A -o wide
kubectl logs -n application deploy/mlflow --tail=200
```

## Автор

Андрій Ілін
