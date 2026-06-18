import json
import sys

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import config
from src.utils.exception import SpamDetectionException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Trains and tunes multiple classifiers, evaluates them on the test
    set, and persists the best-performing model.
    """

    def __init__(self):
        self.models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=config.RANDOM_STATE
            ),
            "Linear SVM": LinearSVC(random_state=config.RANDOM_STATE),
        }

    def _tune_model(self, name, model, X_train, y_train):
        param_grid = config.MODEL_PARAM_GRID.get(name, {})
        if not param_grid:
            model.fit(X_train, y_train)
            return model

        logger.info(f"Running GridSearchCV for {name} with grid {param_grid}")
        search = GridSearchCV(
            model, param_grid, cv=config.CV_FOLDS, scoring="f1", n_jobs=-1
        )
        search.fit(X_train, y_train)
        logger.info(f"{name} best params: {search.best_params_}")
        return search.best_estimator_

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        try:
            results = {}
            fitted_models = {}

            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                best_model = self._tune_model(name, model, X_train, y_train)
                fitted_models[name] = best_model

                y_pred = best_model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                }
                results[name] = metrics

                logger.info(
                    f"{name} -> accuracy={metrics['accuracy']:.4f}, "
                    f"precision={metrics['precision']:.4f}, "
                    f"recall={metrics['recall']:.4f}, "
                    f"f1={metrics['f1']:.4f}"
                )
                logger.info(
                    "\n" + classification_report(
                        y_test, y_pred, target_names=["ham", "spam"]
                    )
                )

            best_model_name = max(results, key=lambda k: results[k]["f1"])
            best_model = fitted_models[best_model_name]

            logger.info(
                f"Best model: {best_model_name} "
                f"(F1 = {results[best_model_name]['f1']:.4f})"
            )

            joblib.dump(best_model, config.MODEL_PATH)
            logger.info(f"Best model saved to {config.MODEL_PATH}")

            with open(config.METRICS_PATH, "w") as f:
                json.dump(
                    {"results": results, "best_model": best_model_name},
                    f,
                    indent=4,
                )
            logger.info(f"Metrics saved to {config.METRICS_PATH}")

            return best_model_name, best_model, results, fitted_models

        except Exception as e:
            raise SpamDetectionException(e, sys)
