import os
import sys

from src.config import config
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.exception import SpamDetectionException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training():
    pipeline = TrainingPipeline()
    pipeline.run()


def run_single_prediction():
    if not os.path.exists(config.MODEL_PATH):
        print("\nNo trained model found. Please run training first (option 1).")
        return

    pipeline = PredictionPipeline()
    text = input("\nEnter a message to classify: ").strip()
    if not text:
        print("Empty message, nothing to classify.")
        return

    result = pipeline.predict_single(text)
    label = result["label"].upper()
    confidence = result["confidence"]

    print(f"\nPrediction: {label}")
    if confidence is not None:
        print(f"Confidence: {confidence:.2%}")


def run_batch_prediction():
    if not os.path.exists(config.MODEL_PATH):
        print("\nNo trained model found. Please run training first (option 1).")
        return

    input_path = input("\nPath to input CSV file: ").strip()
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    text_column = input(
        "Name of the text column (press Enter for 'text'): "
    ).strip() or "text"

    output_path = os.path.join(config.OUTPUTS_DIR, "batch_predictions.csv")

    pipeline = PredictionPipeline()
    pipeline.predict_batch_csv(input_path, output_path, text_column=text_column)

    print(f"\nBatch predictions saved to: {output_path}")


def main_menu():
    while True:
        print("\n" + "=" * 60)
        print("SPAM MESSAGE CLASSIFICATION")
        print("=" * 60)
        print("1. Train models (run full training pipeline)")
        print("2. Classify a single message")
        print("3. Classify messages from a CSV file (batch)")
        print("4. Exit")

        choice = input("\nSelect an option (1-4): ").strip()

        try:
            if choice == "1":
                run_training()
            elif choice == "2":
                run_single_prediction()
            elif choice == "3":
                run_batch_prediction()
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid option, please choose 1-4.")
        except SpamDetectionException as e:
            print(f"\nAn error occurred: {e}")
        except Exception as e:
            print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main_menu()
