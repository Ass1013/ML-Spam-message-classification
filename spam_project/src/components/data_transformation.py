import re
import string
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import config
from src.utils.exception import SpamDetectionException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DataTransformation:
    """Cleans raw text and converts it into TF-IDF feature vectors."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(**config.TFIDF_PARAMS)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["target"] = df["label"].map({"ham": 0, "spam": 1})
        df["clean_text"] = df["text"].apply(clean_text)
        return df

    def fit_transform_train(self, train_df: pd.DataFrame):
        try:
            logger.info("Cleaning training text and fitting TF-IDF vectorizer.")
            train_df = self._prepare_dataframe(train_df)

            X_train = self.vectorizer.fit_transform(train_df["clean_text"])
            y_train = train_df["target"].values

            joblib.dump(self.vectorizer, config.VECTORIZER_PATH)
            logger.info(f"Vectorizer saved to {config.VECTORIZER_PATH}")

            return X_train, y_train

        except Exception as e:
            raise SpamDetectionException(e, sys)

    def transform_test(self, test_df: pd.DataFrame):
        try:
            logger.info("Cleaning test text and applying fitted vectorizer.")
            test_df = self._prepare_dataframe(test_df)

            X_test = self.vectorizer.transform(test_df["clean_text"])
            y_test = test_df["target"].values

            return X_test, y_test

        except Exception as e:
            raise SpamDetectionException(e, sys)

    def run(self, train_path: str = config.TRAIN_DATA_PATH,
            test_path: str = config.TEST_DATA_PATH):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train, y_train = self.fit_transform_train(train_df)
        X_test, y_test = self.transform_test(test_df)

        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    transformation = DataTransformation()
    transformation.run()
