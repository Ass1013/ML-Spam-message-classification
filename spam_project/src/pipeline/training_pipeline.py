import sys

import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.visualizer import Visualizer
from src.config import config
from src.utils.exception import SpamDetectionException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """End-to-end pipeline: ingestion -> transformation -> training -> evaluation."""

    def __init__(self):
        self.ingestion = DataIngestion()
        self.transformation = DataTransformation()
        self.trainer = ModelTrainer()

    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("STARTING TRAINING PIPELINE")
            logger.info("=" * 60)

            train_path, test_path = self.ingestion.run()

            full_df = pd.read_csv(config.RAW_DATASET_PATH, encoding="latin-1")
            full_df = full_df.iloc[:, :2]
            full_df.columns = ["label", "text"]
            full_df = full_df.dropna(subset=["text"]).reset_index(drop=True)
            Visualizer.plot_eda(full_df)

            X_train, y_train, X_test, y_test = self.transformation.run(
                train_path, test_path
            )

            best_model_name, best_model, results, fitted_models = (
                self.trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
            )

            Visualizer.plot_model_comparison(
                results, y_test, fitted_models, X_test, best_model_name
            )

            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE COMPLETE")
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Model saved to: {config.MODEL_PATH}")
            logger.info(f"Vectorizer saved to: {config.VECTORIZER_PATH}")
            logger.info("=" * 60)

            return best_model_name, results

        except Exception as e:
            raise SpamDetectionException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
