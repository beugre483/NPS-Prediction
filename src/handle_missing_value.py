import logging
from abc import ABC, abstractmethod

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): the dataset containing missing values

        Returns:
        pd.DataFrame: the dataset with missing values handled
        """
        pass

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        """
        Parameters:
        axis (int): 0 → drop rows, 1 → drop columns
        thresh (int): minimum number of non-NA values required to keep row/column
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Dropping missing values with axis={self.axis} and thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df_cleaned


class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        """
        Parameters:
        method (str): 'mean', 'median', 'mode', or 'constant'
        fill_value (any): value to use when method='constant'
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Filling missing values using method: {self.method}")
        df_cleaned = df.copy()

        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned


class CategoricalFillStrategy(MissingValueHandlingStrategy):
    def __init__(self, fill_map: dict):
        """
        Fill missing values in specific categorical/ordinal columns
        with a specific value per column.

        Parameters:
        fill_map (dict): mapping of column → fill value
                         example: {"Offer": "No Offer", "Internet Type": "No Internet"}
        """
        self.fill_map = fill_map

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in each column specified in fill_map
        with its corresponding value.

        Parameters:
        df (pd.DataFrame): the dataset containing missing values

        Returns:
        pd.DataFrame: the dataset with missing values filled
        """
        logging.info(f"Filling categorical missing values with map: {self.fill_map}")
        df_cleaned = df.copy()

        for column, fill_value in self.fill_map.items():
            if column not in df_cleaned.columns:
                logging.warning(f"Column '{column}' not found in dataframe — skipped.")
                continue
            missing_count = df_cleaned[column].isna().sum()
            df_cleaned[column] = df_cleaned[column].fillna(fill_value)
            logging.info(f"Column '{column}' — {missing_count} missing values filled with '{fill_value}'.")

        logging.info("Categorical missing values filled.")
        return df_cleaned


# ─── Context ───
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        """
        Parameters:
        strategy (MissingValueHandlingStrategy): the strategy to use
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the missing value handling strategy.

        Parameters:
        df (pd.DataFrame): the dataset

        Returns:
        pd.DataFrame: the dataset with missing values handled
        """
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)