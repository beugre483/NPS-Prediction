import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



# Colonnes à supprimer avant le split
# Appelé sur df_enriched AVANT DataSplitter


COLS_TO_DROP = [
    # Colonnes techniques / sans variance
    "Customer ID",
    "Count",
    "Country",       # toujours "United States"
    "State",         # toujours "California"
    "Lat Long",      # redondant avec Latitude et Longitude
    "Quarter",       # toujours "Q3"
    "ID",      
    "City"# clé technique table population

    # Redondances
    "Churn Label",   # redondant avec Churn Value

    "Churn Score",
    "Churn Value",
    "Churn Category",
    "Churn Reason",
    "Customer Status",
    "CLTV",
    "Zip Code",                           
   "Latitude",                   
   "Longitude",
]



# Colonnes binaires Yes/No + flags créés en FE
# → OrdinalEncoder (0/1)


BINARY_COLS = [
    # Colonnes brutes Yes/No
    "Gender",
    "Senior Citizen",
    "Married",
    "Dependents",

    "Referred a Friend",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
    "Paperless Billing",

    # Features créées dans feature_engineering.py — déjà en 0/1
    "Has_Been_Refunded",          # ValueFeatures
    "Support_Profile",            # EngagementFeatures
    "Heavy_Internet_User",        # EngagementFeatures
    "Contract_Risk",              # ContractFeatures
    "Contract_Tenure_Mismatch",   # ContractFeatures
    "Vulnerable_Senior",          # DemographicFeatures
    "Family_Profile",             # DemographicFeatures
    "Digital_Profile",            # DemographicFeatures
    "High_Value_At_Risk",         # DemographicFeatures
]



# Colonnes catégorielles nominales
# → OneHotEncoder


CATEGORICAL_COLS = [
    "Offer",
    "Internet Type",
    "Payment Method"
    # "Contract" → encodé dans Contract_Encoded (OrdinalEncoder)
]


# Colonnes numériques continues
# → StandardScaler (trees/linear) ou MinMaxScaler (neural net)


NUMERICAL_COLS = [
    # Colonnes brutes
    "Age",
    "Number of Referrals",
    "Tenure in Months",
    "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download",
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Revenue",
    "Population",
    "Number of Dependents",

    # Features créées dans feature_engineering.py
    "Revenue_Per_Month",
    "Charge_Evolution",
    "Refund_Rate",
    "Referral_Rate",
    "Internet_Value",
    "Engagement_Score",
    "Locked_Value",
]


# Colonnes ordinales (ordre implicite)
# → OrdinalEncoder (trees) ou Scaler (linear/neural net)

ORDINAL_COLS = [
    "Contract_Encoded",          # 0=M2M, 1=1Y, 2=2Y
    "Tenure_Segment",            # 0=Nouveau … 3=Fidèle
    "Services_Count",            # 0 à 9
    "Streaming_Count",           # 0 à 3
    "Security_Score",            # 0 à 4
    "Full_Protection_Score",     # 0 à 4
    "Service_Diversity_Score",   # 0 à 6
    "Billing_Complexity",        # 0 à 3
]



# Abstract Base Class — Template Pattern

class PreprocessingTemplate(ABC):
    """
    Template Pattern — squelette fixe du preprocessing.

    Ordre d'appel dans le notebook :
    1. feature_engineering  → df_enriched
    2. preprocessing        → drop colonnes inutiles → df_clean
    3. DataSplitter         → sépare X/y, puis répondants/silencieux, puis train/val/test
    4. sklearn Pipeline     → ColumnTransformer.fit(X_train) → transform(X_val, X_test)

    Ce que fait preprocess() :
    - Dropper les colonnes inutiles et données futures
    - Retourner le ColumnTransformer prêt à être intégré dans la Pipeline sklearn

    Ce qui change selon le modèle :
    - Le scaler (StandardScaler vs MinMaxScaler)
    - Le traitement des ordinales (OrdinalEncoder vs Scaler)
    """

    def preprocess(self, X: pd.DataFrame):
        """
        Drop colonnes inutiles et construire le ColumnTransformer.

        À appeler sur X avant le DataSplitter — uniquement sur les features,
        pas sur le dataset complet avec la target.

        Parameters:
        X (pd.DataFrame): les features — dataset enrichi sans la target

        Returns:
        X_clean (pd.DataFrame): features nettoyées — prêtes pour le DataSplitter
        preprocessor (ColumnTransformer): à intégrer dans la sklearn Pipeline
        """
        logging.info("Starting preprocessing pipeline.")

        X_clean = self._drop_columns(X)
        preprocessor = self._build_preprocessor(X_clean)

        logging.info("Preprocessing pipeline ready.")
        return X_clean, preprocessor

    def drop_from_full_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Variante pour dropper les colonnes inutiles sur le dataset complet
        (avec la target encore présente) avant le DataSplitter.

        Parameters:
        df (pd.DataFrame): dataset enrichi complet (features + target)

        Returns:
        pd.DataFrame: dataset nettoyé avec la target toujours présente
        """
        return self._drop_columns(df)

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop technical, redundant and future-data columns.

        Parameters:
        df (pd.DataFrame): the dataset

        Returns:
        pd.DataFrame: cleaned dataset
        """
        cols_present = [col for col in COLS_TO_DROP if col in df.columns]
        logging.info(f"Dropping {len(cols_present)} columns: {cols_present}")
        return df.drop(columns=cols_present)

    def _get_binary_cols(self, X: pd.DataFrame) -> list:
        return [col for col in BINARY_COLS if col in X.columns]

    def _get_categorical_cols(self, X: pd.DataFrame) -> list:
        return [col for col in CATEGORICAL_COLS if col in X.columns]

    def _get_numerical_cols(self, X: pd.DataFrame) -> list:
        return [col for col in NUMERICAL_COLS if col in X.columns]

    def _get_ordinal_cols(self, X: pd.DataFrame) -> list:
        return [col for col in ORDINAL_COLS if col in X.columns]

    @abstractmethod
    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Build and return the ColumnTransformer.
        Implemented differently per model family.

        Parameters:
        X (pd.DataFrame): the feature dataframe

        Returns:
        ColumnTransformer
        """
        pass


# Concrete Class 1 — Tree-Based Models
# XGBoost, Random Forest, LightGBM, CatBoost, Gradient Boosting


class TreeBasedPreprocessor(PreprocessingTemplate):
    """
    Preprocessor for tree-based models.
    Covers: XGBoost, Random Forest, LightGBM, CatBoost, Gradient Boosting.

    Trees split on thresholds — not sensitive to feature scale.
    StandardScaler applied for pipeline consistency but does not impact performance.
    Ordinal features encoded with OrdinalEncoder to preserve natural order.
    """

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Numerical  : StandardScaler  — not critical but consistent
        Binary     : OrdinalEncoder  — Yes/No → 1/0
        Ordinal    : OrdinalEncoder  — preserves natural order
        Categorical: OneHotEncoder   — dummy variables
        """
        binary_cols      = self._get_binary_cols(X)
        categorical_cols = self._get_categorical_cols(X)
        numerical_cols   = self._get_numerical_cols(X)
        ordinal_cols     = self._get_ordinal_cols(X)

        logging.info(f"[TreeBased] {len(numerical_cols)} numerical, {len(ordinal_cols)} ordinal, "
                     f"{len(binary_cols)} binary, {len(categorical_cols)} categorical cols.")

        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(),                               numerical_cols),
                ("bin", OrdinalEncoder(),                               binary_cols),
                ("ord", OrdinalEncoder(),                               ordinal_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False),             categorical_cols),
            ],
            remainder="drop"
        )



# Concrete Class 2 — SVM + Logistic Regression + KNN


class LinearPreprocessor(PreprocessingTemplate):
    """
    Preprocessor for distance/margin based models.
    Covers: SVM, Logistic Regression, KNN.

    Very sensitive to feature scale — StandardScaler is critical
    for all numerical AND ordinal features.
    """

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Numerical  : StandardScaler — CRITICAL
        Ordinal    : StandardScaler — also scaled
        Binary     : OrdinalEncoder — Yes/No → 1/0
        Categorical: OneHotEncoder  — dummy variables
        """
        binary_cols      = self._get_binary_cols(X)
        categorical_cols = self._get_categorical_cols(X)
        numerical_cols   = self._get_numerical_cols(X)
        ordinal_cols     = self._get_ordinal_cols(X)

        logging.info(f"[Linear] {len(numerical_cols)} numerical, {len(ordinal_cols)} ordinal, "
                     f"{len(binary_cols)} binary, {len(categorical_cols)} categorical cols.")

        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(),                               numerical_cols),
                ("ord", StandardScaler(),                               ordinal_cols),
                ("bin", OrdinalEncoder(),                               binary_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False),             categorical_cols),
            ],
            remainder="drop"
        )


# ─────────────────────────────────────────────
# Concrete Class 3 — Neural Network / MLP
# ─────────────────────────────────────────────

class NeuralNetPreprocessor(PreprocessingTemplate):
    """
    Preprocessor for Neural Networks (MLP, Deep Learning).

    All features must be in [0, 1] — MinMaxScaler applied to numerical and ordinal.
    Binary and OneHot outputs are already in {0, 1}.
    """

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Numerical  : MinMaxScaler  — maps to [0, 1]
        Ordinal    : MinMaxScaler  — also mapped to [0, 1]
        Binary     : OrdinalEncoder — Yes/No → 1/0, already in {0,1}
        Categorical: OneHotEncoder  — already in {0,1}
        """
        binary_cols      = self._get_binary_cols(X)
        categorical_cols = self._get_categorical_cols(X)
        numerical_cols   = self._get_numerical_cols(X)
        ordinal_cols     = self._get_ordinal_cols(X)

        logging.info(f"[NeuralNet] {len(numerical_cols)} numerical, {len(ordinal_cols)} ordinal, "
                     f"{len(binary_cols)} binary, {len(categorical_cols)} categorical cols.")

        return ColumnTransformer(
            transformers=[
                ("num", MinMaxScaler(),                                 numerical_cols),
                ("ord", MinMaxScaler(),                                 ordinal_cols),
                ("bin", OrdinalEncoder(),                               binary_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False),             categorical_cols),
            ],
            remainder="drop"
        )