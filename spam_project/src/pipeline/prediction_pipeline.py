import sys

import joblib
import pandas as pd
from src.components.data_transformation import clean_text
from src.config import config
from src.utils.exception import SpamDetectionException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionPipeline:
    """Loads the trained model and vectorizer to classify new messages."""

    def __init__(self, model_path: str = config.MODEL_PATH,
                 vectorizer_path: str = config.VECTORIZER_PATH):
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)

            logger.info(f"Loading vectorizer from {vectorizer_path}")
            self.vectorizer = joblib.load(vectorizer_path)

        except Exception as e:
            raise SpamDetectionException(e, sys)

    def predict_single(self, text: str) -> dict:
        try:
            cleaned = clean_text(text)
            vec = self.vectorizer.transform([cleaned])
            prediction = self.model.predict(vec)[0]
            label = "spam" if prediction == 1 else "ham"

            confidence = None
            if hasattr(self.model, "predict_proba"):
                confidence = float(self.model.predict_proba(vec)[0][prediction])
            elif hasattr(self.model, "decision_function"):
                score = self.model.decision_function(vec)[0]
                prob_spam = 1 / (1 + pow(2.718281828, -score))
                confidence = float(prob_spam if prediction == 1 else 1 - prob_spam)

            return {"text": text, "label": label, "confidence": confidence}

        except Exception as e:
            raise SpamDetectionException(e, sys)

    def predict_batch_csv(self, input_csv_path: str, output_csv_path: str,
                           text_column: str = "text") -> str:
        try:
            logger.info(f"Loading batch input from {input_csv_path}")
            df = pd.read_csv(input_csv_path, encoding="latin-1")

            if text_column not in df.columns:
                raise ValueError(
                    f"Column '{text_column}' not found in {input_csv_path}. "
                    f"Available columns: {list(df.columns)}"
                )

            df["clean_text"] = df[text_column].astype(str).apply(clean_text)
            vec = self.vectorizer.transform(df["clean_text"])
            predictions = self.model.predict(vec)

            df["prediction"] = ["spam" if p == 1 else "ham" for p in predictions]
            df = df.drop(columns=["clean_text"])

            df.to_csv(output_csv_path, index=False)
            logger.info(f"Batch predictions saved to {output_csv_path}")

            return output_csv_path

        except Exception as e:
            raise SpamDetectionException(e, sys)


if __name__ == "__main__":
    pipeline = PredictionPipeline()

    samples = [
        "Congratulations! You have won a free ticket to Bahamas. Call now!",
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
    ]
    for text in samples:
        result = pipeline.predict_single(text)
        print(result)
