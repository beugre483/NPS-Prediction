import logging
from abc import ABC, abstractmethod

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ─── Abstract Base Class ───
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from the existing dataset.

        Parameters:
        df (pd.DataFrame): the dataset

        Returns:
        pd.DataFrame: the dataset with new features added
        """
        pass


# ─── Concrete Strategies ───

class ValueFeatures(FeatureEngineeringStrategy):
    """
    Features liées à la valeur financière du client.

    Features créées :
    - Revenue_Per_Month     : revenu moyen mensuel réel du client
    - Charge_Evolution      : écart entre tarif actuel et historique → upgrade ou downgrade
    - Refund_Rate           : taux de remboursement → proxy d'insatisfaction
    - Has_Been_Refunded     : flag — le client a-t-il déjà été remboursé ?
    - Billing_Complexity    : nombre de types de charges supplémentaires
    """

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Creating value-based features.")
        df = df.copy()

        df["Revenue_Per_Month"] = df["Total Revenue"] / df["Tenure in Months"]

        df["Charge_Evolution"] = df["Monthly Charge"] - df["Revenue_Per_Month"]

        df["Refund_Rate"] = df["Total Refunds"] / df["Total Revenue"]

        df["Has_Been_Refunded"] = (df["Total Refunds"] > 0).astype(int)

        df["Billing_Complexity"] = (
            (df["Total Extra Data Charges"] > 0).astype(int) +
            (df["Total Long Distance Charges"] > 0).astype(int) +
            (df["Total Refunds"] > 0).astype(int)
        )

        logging.info("Value-based features created.")
        return df


class EngagementFeatures(FeatureEngineeringStrategy):
    """
    Features liées à l'engagement et aux services souscrits.

    Features créées :
    - Services_Count        : nombre total de services souscrits
    - Streaming_Count       : nombre de services streaming souscrits
    - Security_Score        : nombre de services de sécurité/protection souscrits
    - Full_Protection_Score : score global de protection (0 à 4)
    - Support_Profile       : client avec support premium + protection équipement
    - Service_Diversity_Score: score global d'engagement multi-services
    - Referral_Rate         : taux de recommandation par mois d'ancienneté
    - Heavy_Internet_User   : gros consommateur de data (> 75ème percentile)
    - Internet_Value        : combinaison type internet × usage data
    """

    SERVICES_COLS = [
        "Online Security", "Online Backup", "Device Protection Plan",
        "Premium Tech Support", "Streaming TV", "Streaming Movies",
        "Streaming Music", "Unlimited Data", "Multiple Lines"
    ]

    STREAMING_COLS = ["Streaming TV", "Streaming Movies", "Streaming Music"]

    SECURITY_COLS = [
        "Online Security", "Online Backup",
        "Device Protection Plan", "Premium Tech Support"
    ]

    INTERNET_TYPE_MAP = {"DSL": 1, "Cable": 2, "Fiber Optic": 3}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Creating engagement-based features.")
        df = df.copy()

        df["Services_Count"] = df[self.SERVICES_COLS].apply(
            lambda x: (x == "Yes").sum(), axis=1
        )

        df["Streaming_Count"] = df[self.STREAMING_COLS].apply(
            lambda x: (x == "Yes").sum(), axis=1
        )

        df["Security_Score"] = df[self.SECURITY_COLS].apply(
            lambda x: (x == "Yes").sum(), axis=1
        )

        protection_cols = ["Online Security", "Online Backup",
                           "Device Protection Plan", "Premium Tech Support"]
        df["Full_Protection_Score"] = df[protection_cols].apply(
            lambda x: (x == "Yes").sum(), axis=1
        )

        df["Support_Profile"] = (
            (df["Premium Tech Support"] == "Yes") &
            (df["Device Protection Plan"] == "Yes")
        ).astype(int)

        df["Service_Diversity_Score"] = (
            df["Streaming_Count"] +
            df["Full_Protection_Score"] +
            (df["Multiple Lines"] == "Yes").astype(int) +
            (df["Unlimited Data"] == "Yes").astype(int)
        )

        df["Referral_Rate"] = df["Number of Referrals"] / df["Tenure in Months"]

        df["Heavy_Internet_User"] = (
            df["Avg Monthly GB Download"] > df["Avg Monthly GB Download"].quantile(0.75)
        ).astype(int)

        df["Internet_Value"] = (
            df["Internet Type"].map(self.INTERNET_TYPE_MAP).fillna(0) *
            df["Avg Monthly GB Download"]
        )

        logging.info("Engagement-based features created.")
        return df


class ContractFeatures(FeatureEngineeringStrategy):
    """
    Features liées au contrat et à l'engagement contractuel.

    Features créées :
    - Contract_Encoded          : encodage ordinal du contrat (0, 1, 2)
    - Engagement_Score          : contrat × ancienneté → engagement réel
    - Contract_Risk             : nouveau client sans engagement → risque maximum
    - Contract_Tenure_Mismatch  : client fidèle toujours sans engagement → profil suspect
    - Locked_Value              : contrat × services → difficulté à partir
    """

    CONTRACT_MAP = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Creating contract-based features.")
        df = df.copy()

        df["Contract_Encoded"] = df["Contract"].map(self.CONTRACT_MAP)

        df["Engagement_Score"] = df["Contract_Encoded"] * df["Tenure in Months"]

        df["Contract_Risk"] = (
            (df["Contract"] == "Month-to-Month") &
            (df["Tenure in Months"] < 12)
        ).astype(int)

        df["Contract_Tenure_Mismatch"] = (
            (df["Contract"] == "Month-to-Month") &
            (df["Tenure in Months"] > 24)
        ).astype(int)

        # Nécessite Services_Count — à appliquer après EngagementFeatures
        if "Services_Count" in df.columns:
            df["Locked_Value"] = df["Contract_Encoded"] * df["Services_Count"]
        else:
            logging.warning(
                "Services_Count not found — apply EngagementFeatures before ContractFeatures "
                "to compute Locked_Value."
            )

        logging.info("Contract-based features created.")
        return df


class TenureFeatures(FeatureEngineeringStrategy):
    """
    Features liées à l'ancienneté du client.

    Features créées :
    - Tenure_Segment        : segmentation de l'ancienneté en 4 groupes ordonnés
                              0 = Nouveau (0-12 mois)
                              1 = Développement (12-24 mois)
                              2 = Etabli (24-48 mois)
                              3 = Fidèle (48-72 mois)
    """

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Creating tenure-based features.")
        df = df.copy()

        df["Tenure_Segment"] = pd.cut(
            df["Tenure in Months"],
            bins=[0, 12, 24, 48, 72],
            labels=[0, 1, 2, 3]
        ).astype(int)

        logging.info("Tenure-based features created.")
        return df


class DemographicFeatures(FeatureEngineeringStrategy):
    """
    Features liées au profil démographique du client.

    Features créées :
    - Vulnerable_Senior : senior sans support premium → risque d'insatisfaction technique
    - Family_Profile    : client marié avec personnes à charge → besoins spécifiques
    - Digital_Profile   : client full digital → paperless + prélèvement + internet
    - High_Value_At_Risk: client à haute valeur CLTV, récent et sans engagement
    """

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Creating demographic-based features.")
        df = df.copy()

        df["Vulnerable_Senior"] = (
            (df["Senior Citizen"] == "Yes") &
            (df["Premium Tech Support"] == "No")
        ).astype(int)

        df["Family_Profile"] = (
            (df["Married"] == "Yes") &
            (df["Dependents"] == "Yes")
        ).astype(int)

        df["Digital_Profile"] = (
            (df["Paperless Billing"] == "Yes") &
            (df["Payment Method"] == "Bank Withdrawal") &
            (df["Internet Service"] == "Yes")
        ).astype(int)

        df["High_Value_At_Risk"] = (
            (df["CLTV"] > df["CLTV"].quantile(0.75)) &
            (df["Tenure in Months"] < 12) &
            (df["Contract"] == "Month-to-Month")
        ).astype(int)

        logging.info("Demographic-based features created.")
        return df
    

class NPSLabelFeature(FeatureEngineeringStrategy):
    """
    Dérive la classe NPS à partir du Satisfaction Score (1→5).

    Justification du mapping :
    - Score 5 → Promoter  : client très satisfait, susceptible de recommander
    - Score 4 → Passive   : client satisfait mais pas enthousiaste
    - Score ≤ 3 → Detractor : client insatisfait, risque de churn et bouche-à-oreille négatif

    La colonne originale Satisfaction Score est conservée — 
    NPS_Label devient la nouvelle target du modèle.

    Encodage :
    0 = Detractor
    1 = Passive
    2 = Promoter
    """

    NPS_MAP = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2}
    NPS_LABELS = {0: "Detractor", 1: "Passive", 2: "Promoter"}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Creating NPS label from Satisfaction Score.")
        df = df.copy()

        df["NPS_Label"] = df["Satisfaction Score"].map(self.NPS_MAP)

        # Log de la distribution pour vérification
        dist = df["NPS_Label"].map(self.NPS_LABELS).value_counts(normalize=True).round(3)
        logging.info(f"NPS distribution :\n{dist}")

        logging.info("NPS label created.")
        return df
    
    


# ─── Context ───
class FeatureEngineer:
    def __init__(self, strategies: list):
        """
        Initializes the FeatureEngineer with a list of strategies to apply.

        Parameters:
        strategies (list): list of FeatureEngineeringStrategy instances
                           applied in the order provided
        """
        self.strategies = strategies

    def set_strategies(self, strategies: list):
        """
        Replace the current list of strategies.

        Parameters:
        strategies (list): new list of FeatureEngineeringStrategy instances
        """
        logging.info("Updating feature engineering strategies.")
        self.strategies = strategies

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all strategies in order to the dataset.

        Parameters:
        df (pd.DataFrame): the dataset

        Returns:
        pd.DataFrame: the enriched dataset with all new features
        """
        logging.info(f"Running {len(self.strategies)} feature engineering strategies.")
        for strategy in self.strategies:
            df = strategy.create_features(df)
        logging.info("All feature engineering strategies completed.")
        return df