import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import config
from src.utils.exception import SpamDetectionException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """Loads the raw dataset, keeps only the relevant columns, and splits
    it into train / test CSV files used by the rest of the pipeline.
    """

    def __init__(self, raw_path: str = config.RAW_DATASET_PATH):
        self.raw_path = raw_path

    def load_raw_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading raw dataset from {self.raw_path}")
            try:
                df = pd.read_csv(self.raw_path, encoding="latin-1")
            except pd.errors.ParserError:
                logger.warning(
                    "Malformed rows detected in the CSV file, "
                    "retrying with a tolerant parser and skipping bad lines."
                )
                df = pd.read_csv(
                    self.raw_path,
                    encoding="latin-1",
                    engine="python",
                    sep=",",
                    quotechar='"',
                    on_bad_lines="skip",
                )

            df = df.iloc[:, :2]
            df.columns = ["label", "text"]
            df = df.dropna(subset=["text"]).reset_index(drop=True)
            df = df.drop_duplicates().reset_index(drop=True)

            logger.info(f"Loaded {len(df)} rows after cleaning.")
            return df

        except Exception as e:
            raise SpamDetectionException(e, sys)

    def split_data(self, df: pd.DataFrame):
        try:
            logger.info("Splitting dataset into train and test sets.")

            train_df, test_df = train_test_split(
                df,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE,
                stratify=df["label"],
            )

            train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
            test_df.to_csv(config.TEST_DATA_PATH, index=False)

            logger.info(
                f"Train set: {len(train_df)} rows -> {config.TRAIN_DATA_PATH}"
            )
            logger.info(
                f"Test set: {len(test_df)} rows -> {config.TEST_DATA_PATH}"
            )

            return config.TRAIN_DATA_PATH, config.TEST_DATA_PATH

        except Exception as e:
            raise SpamDetectionException(e, sys)

    def run(self):
        df = self.load_raw_data()
        return self.split_data(df)


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()
