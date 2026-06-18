import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

RAW_DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

MODEL_PATH = os.path.join(OUTPUTS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(OUTPUTS_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics.json")
EDA_PLOT_PATH = os.path.join(OUTPUTS_DIR, "eda_overview.png")
COMPARISON_PLOT_PATH = os.path.join(OUTPUTS_DIR, "model_comparison.png")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
}

MODEL_PARAM_GRID = {
    "Naive Bayes": {
        "alpha": [0.1, 0.5, 1.0],
    },
    "Logistic Regression": {
        "C": [0.1, 1.0, 10.0],
    },
    "Linear SVM": {
        "C": [0.1, 1.0, 10.0],
    },
}

for _dir in (DATA_DIR, OUTPUTS_DIR, LOGS_DIR):
    os.makedirs(_dir, exist_ok=True)
