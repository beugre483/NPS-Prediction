import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ─── Abstract Base Class ───
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Split the dataset according to the strategy.

        Parameters:
        df (pd.DataFrame): the dataset
        target_column (str): the target column name

        Returns:
        splits according to the chosen strategy
        """
        pass


# ─── Concrete Strategy 1 — Simulation répondants ───
class RespondentSimulationSplit(DataSplittingStrategy):
    def __init__(
        self,
        response_rate: float = 0.15,
        random_state: int = 42
    ):
        """
        Sépare uniquement les répondants des silencieux.
        Stratifié sur le target.

        Parameters:
        response_rate (float): taux de répondants — default 15%
        random_state (int): seed
        """
        self.response_rate = response_rate
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Sépare le dataset en deux groupes :
        - Répondants (15%) → ceux sur lesquels on va travailler
        - Silencieux (85%) → non utilisés pour l'entraînement

        Les deux groupes sont stratifiés sur le Satisfaction Score.

        Parameters:
        df (pd.DataFrame): le dataset complet
        target_column (str): la colonne cible

        Returns:
        df_respondents, df_silent
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]

        logging.info(
            f"Simulating {self.response_rate * 100}% respondents "
            f"stratified on '{target_column}'."
        )

        # Stratifié sur y — sklearn garantit la même distribution dans les deux groupes
        X_silent, X_respondents, y_silent, y_respondents = train_test_split(
            X, y,
            test_size=self.response_rate,
            random_state=self.random_state,
            stratify=y
        )

        df_respondents = pd.concat([X_respondents, y_respondents], axis=1)
        df_silent = pd.concat([X_silent, y_silent], axis=1)

        logging.info(f"Dataset complet : {len(df)} clients")
        logging.info(f"Répondants      : {len(df_respondents)} clients ({self.response_rate * 100}%)")
        logging.info(f"Silencieux      : {len(df_silent)} clients ({(1 - self.response_rate) * 100}%)")
        logging.info(f"Distribution répondants : {y_respondents.value_counts(normalize=True).round(3).to_dict()}")
        logging.info(f"Distribution silencieux : {y_silent.value_counts(normalize=True).round(3).to_dict()}")

        return df_respondents, df_silent


# ─── Concrete Strategy 2 — Train / Val / Test sur les répondants ───
class TrainValTestSplit(DataSplittingStrategy):
    def __init__(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        """
        Split Train / Validation / Test sur les répondants uniquement.
        Tout est stratifié sur le Satisfaction Score.

        Parameters:
        test_size (float): proportion du test set — default 15%
        val_size (float): proportion du validation set — default 15%
        random_state (int): seed
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Split stratifié Train / Validation / Test sur les répondants.

        Parameters:
        df (pd.DataFrame): les répondants uniquement (15% du dataset complet)
        target_column (str): la colonne cible

        Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # ── Étape 1 — Séparer test d'abord ──
        logging.info(f"Splitting test set ({self.test_size * 100}%) stratified on '{target_column}'.")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # ── Étape 2 — Séparer validation depuis le reste ──
        val_size_adjusted = self.val_size / (1 - self.test_size)

        logging.info(f"Splitting val set ({self.val_size * 100}%) stratified on '{target_column}'.")

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )

        logging.info(f"Train      : {len(X_train)} clients ({len(X_train) / len(df) * 100:.1f}%)")
        logging.info(f"Validation : {len(X_val)} clients ({len(X_val) / len(df) * 100:.1f}%)")
        logging.info(f"Test       : {len(X_test)} clients ({len(X_test) / len(df) * 100:.1f}%)")
        logging.info(f"Distribution Train      : {y_train.value_counts(normalize=True).round(3).to_dict()}")
        logging.info(f"Distribution Validation : {y_val.value_counts(normalize=True).round(3).to_dict()}")
        logging.info(f"Distribution Test       : {y_test.value_counts(normalize=True).round(3).to_dict()}")

        return X_train, X_val, X_test, y_train, y_val, y_test


# ─── Context ───
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Parameters:
        strategy (DataSplittingStrategy): the strategy to use
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Parameters:
        strategy (DataSplittingStrategy): the new strategy to use
        """
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        """
        Execute the splitting strategy.

        Parameters:
        df (pd.DataFrame): the dataset
        target_column (str): the target column name

        Returns:
        splits according to the chosen strategy
        """
        logging.info("Executing data splitting strategy.")
        return self._strategy.split_data(df, target_column)