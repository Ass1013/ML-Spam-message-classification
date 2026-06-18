import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.config import config
from src.utils.exception import SpamDetectionException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Visualizer:
    """Generates EDA plots and model comparison plots."""

    @staticmethod
    def plot_eda(df: pd.DataFrame):
        try:
            df = df.copy()
            df["text_length"] = df["text"].apply(len)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            df["label"].value_counts().plot(
                kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"]
            )
            axes[0].set_title("Class distribution")
            axes[0].set_xlabel("Class")
            axes[0].set_ylabel("Number of messages")

            sns.histplot(
                data=df, x="text_length", hue="label", bins=50,
                kde=True, ax=axes[1], palette=["#4C72B0", "#DD8452"]
            )
            axes[1].set_title("Message length: spam vs ham")
            axes[1].set_xlabel("Text length (characters)")

            plt.tight_layout()
            plt.savefig(config.EDA_PLOT_PATH, dpi=150)
            plt.close()
            logger.info(f"EDA plot saved to {config.EDA_PLOT_PATH}")

        except Exception as e:
            raise SpamDetectionException(e, sys)

    @staticmethod
    def plot_model_comparison(results, y_test, fitted_models, X_test, best_model_name):
        try:
            fig, axes = plt.subplots(2, 2, figsize=(13, 11))

            metrics_df = pd.DataFrame(results).T[
                ["accuracy", "precision", "recall", "f1"]
            ]
            metrics_df.plot(kind="bar", ax=axes[0, 0])
            axes[0, 0].set_title("Model comparison")
            axes[0, 0].set_ylabel("Score")
            axes[0, 0].set_ylim(0.8, 1.0)
            axes[0, 0].legend(loc="lower right")
            axes[0, 0].tick_params(axis="x", rotation=20)

            best_model = fitted_models[best_model_name]
            y_pred_best = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_best)
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"],
                ax=axes[0, 1]
            )
            axes[0, 1].set_title(f"Confusion matrix: {best_model_name}")
            axes[0, 1].set_xlabel("Predicted")
            axes[0, 1].set_ylabel("Actual")

            for name, model in fitted_models.items():
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(X_test)[:, 1]
                else:
                    scores = model.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, scores)
                roc_auc = auc(fpr, tpr)
                axes[1, 0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

            axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[1, 0].set_title("ROC curves")
            axes[1, 0].set_xlabel("False Positive Rate")
            axes[1, 0].set_ylabel("True Positive Rate")
            axes[1, 0].legend(loc="lower right")

            names = list(results.keys())
            f1s = [results[n]["f1"] for n in names]
            axes[1, 1].bar(names, f1s, color="#55A868")
            axes[1, 1].set_title("F1-score by model")
            axes[1, 1].set_ylabel("F1-score")
            axes[1, 1].set_ylim(0.8, 1.0)
            axes[1, 1].tick_params(axis="x", rotation=20)

            plt.tight_layout()
            plt.savefig(config.COMPARISON_PLOT_PATH, dpi=150)
            plt.close()
            logger.info(f"Model comparison plot saved to {config.COMPARISON_PLOT_PATH}")

        except Exception as e:
            raise SpamDetectionException(e, sys)
