import os
from pathlib import Path

RANDOM_SEED = 42

TARGET_COLUMN = "Churn"
TARGET_ALIASES = ["Churn", "Churn?"]
ID_COLUMNS = ["CustomerID"]

# Permite rodar instalado como pacote, Docker ou diretamente na raiz do projeto.
PROJECT_ROOT = Path(os.getenv("CHURN_PROJECT_ROOT", Path.cwd())).resolve()
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_logreg.joblib"
CHAMPION_MODEL_PATH = MODELS_DIR / "champion.joblib"

# Corte padrao (baseline). O campeao usa o corte salvo no proprio modelo.
PREDICTION_THRESHOLD = 0.5

# MLflow: mesmo backend e experimento usados no notebook.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "churn-cell2cell"
