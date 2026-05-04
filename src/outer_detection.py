import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers in the given DataFrame.

        Parameters:
        df (pd.DataFrame): the dataset — only numerical columns

        Returns:
        pd.DataFrame: boolean dataframe — True where outlier is detected
        """
        pass


# ─── Concrete Strategies ───
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold: int = 3):
        """
        Parameters:
        threshold (int): Z-score threshold above which a value is considered an outlier
                         default = 3
        """
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Detecting outliers using Z-score method with threshold={self.threshold}")
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info("Z-score outlier detection completed.")
        return outliers


class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using IQR method.")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info("IQR outlier detection completed.")
        return outliers


# ─── Context ───
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        """
        Parameters:
        strategy (OutlierDetectionStrategy): the strategy to use for outlier detection
        """
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using the current strategy.
        Only applies on numerical columns.

        Parameters:
        df (pd.DataFrame): the dataset

        Returns:
        pd.DataFrame: boolean dataframe of outliers
        """
        numerical_df = df.select_dtypes(exclude=["object"])
        logging.info("Executing outlier detection strategy.")
        return self._strategy.detect_outliers(numerical_df)


    def visualize_outliers(self, df: pd.DataFrame, features: list = None):
        """
        Visualize outliers using boxplots.
        If no features provided, visualizes all numerical columns.

        Parameters:
        df (pd.DataFrame): the dataset
        features (list): optional - specific features to visualize
        """
        if features is None:
            features = df.select_dtypes(exclude=["object"]).columns.tolist()

        n_cols = 3
        n_rows = len(features) // n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten()

        for i, feature in enumerate(features):
            sns.boxplot(x=df[feature], ax=axes[i])
            axes[i].set_title(f"Boxplot of {feature}")

        for j in range(len(features), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        logging.info("Outlier visualization completed.")

    def get_outlier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a summary of outliers per column —
        number and percentage of outliers detected.

        Parameters:
        df (pd.DataFrame): the dataset

        Returns:
        pd.DataFrame: summary with count and percentage of outliers per column
        """
        numerical_df = df.select_dtypes(exclude=["object"])
        outliers = self._strategy.detect_outliers(numerical_df)

        summary = pd.DataFrame({
            "outlier_count"      : outliers.sum(),
            "outlier_percentage" : (outliers.sum() / len(df) * 100).round(2)
        })

        return summary[summary["outlier_count"] > 0].sort_values(
            "outlier_percentage", ascending=False
        )