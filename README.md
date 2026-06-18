# Spam Message Classification

A modular machine learning pipeline that classifies SMS/email messages as
**spam** or **ham** (legitimate). Built with a clean separation between data
ingestion, transformation, model training, and inference, plus a simple
console interface for running everything.

## Key Features

- **Modular ML pipeline**: separate components for data ingestion, text
  cleaning/vectorization, and model training.
- **Multiple model comparison**: Naive Bayes, Logistic Regression, and
  Linear SVM, each tuned with `GridSearchCV` and evaluated with 5-fold
  cross-validation.
- **Automatic best-model selection**: the model with the highest F1-score
  on the test set is saved and used for inference.
- **Console interface**: train, classify a single message, or classify an
  entire CSV file of messages, all from a simple menu.
- **Batch CSV predictions**: upload a CSV of messages and get a CSV back
  with a `prediction` column added.
- **Logging**: every pipeline run writes a timestamped log file to `logs/`.

## Tech Stack

- **Language**: Python 3.10+
- **ML framework**: scikit-learn
- **Data processing**: pandas, NumPy
- **Visualization**: matplotlib, seaborn

## Project Structure

```
spam_project/
├── main.py                        # Console entry point (menu)
├── requirements.txt
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Load raw CSV, clean, split train/test
│   │   ├── data_transformation.py # Text cleaning + TF-IDF vectorization
│   │   ├── model_trainer.py       # Train/tune/evaluate models, pick best
│   │   └── visualizer.py          # EDA and model comparison plots
│   ├── pipeline/
│   │   ├── training_pipeline.py   # Orchestrates the full training run
│   │   └── prediction_pipeline.py # Loads saved model for inference
│   ├── config/
│   │   └── config.py              # Paths, hyperparameters, constants
│   └── utils/
│       ├── logger.py              # Timestamped file + console logging
│       └── exception.py           # Custom exception with file/line info
├── data/
│   └── dataset.csv                # Raw dataset (label, text columns)
├── outputs/                       # model.pkl, vectorizer.pkl, plots, metrics.json
└── logs/                          # One log file per run
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Train the models

Place your dataset at `data/dataset.csv` with at least two columns: a label
column (`ham`/`spam`) and a text column. Then run:

```bash
python main.py
```

and choose option **1** from the menu. This will:

1. Load and clean the raw dataset.
2. Split it into train/test sets (saved to `data/train.csv` and `data/test.csv`).
3. Clean the text and fit a TF-IDF vectorizer (saved to `outputs/vectorizer.pkl`).
4. Train and tune Naive Bayes, Logistic Regression, and Linear SVM.
5. Evaluate all three models and save the best one to `outputs/model.pkl`.
6. Save evaluation metrics to `outputs/metrics.json`.
7. Save `outputs/eda_overview.png` and `outputs/model_comparison.png`.

You can also run the training pipeline directly without the menu:

```bash
python -m src.pipeline.training_pipeline
```

### 2. Classify a single message

From the menu, choose option **2** and type a message. You'll get back a
`spam`/`ham` label and a confidence score.

### 3. Classify a batch of messages from a CSV

From the menu, choose option **3**, then provide the path to a CSV file
and the name of the column containing the message text. A new file
`outputs/batch_predictions.csv` will be created with a `prediction` column
added.

## Configuration

All paths and hyperparameters live in `src/config/config.py`:

- `RAW_DATASET_PATH`, `TRAIN_DATA_PATH`, `TEST_DATA_PATH`
- `MODEL_PATH`, `VECTORIZER_PATH`, `METRICS_PATH`
- `TFIDF_PARAMS` (max_features, ngram_range, min_df)
- `MODEL_PARAM_GRID` (GridSearchCV grids per model)
- `TEST_SIZE`, `CV_FOLDS`, `RANDOM_STATE`

## Model Performance

The training pipeline evaluates all models with 5-fold cross-validation
during grid search, then reports accuracy, precision, recall, and F1-score
on the held-out test set. Results are printed to the console, logged to
`logs/`, and saved to `outputs/metrics.json`. The model with the best
F1-score is automatically selected for inference.

## License

This project is provided as-is for educational purposes.
