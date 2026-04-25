import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
from abc import ABC, abstractmethod


class UnivariateAnalysisStrategy(ABC):
    
    @abstractmethod
    def analyse(self, data: pd.DataFrame, feature: str):
        """Perform univariate analysis on a single feature
        
        Parameters:
        data (DataFrame): the dataset
        feature (str): the feature to analyse
        """
        pass

    @abstractmethod
    def analyse_all(self, data: pd.DataFrame):
        """Perform univariate analysis on all relevant columns
        
        Parameters:
        data (DataFrame): the dataset
        """
        pass


class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    
    def analyse(self, data: pd.DataFrame, feature: str):
        """Histogramme + KDE pour une seule variable numérique
        
        Parameters:
        data (DataFrame): the dataset
        feature (str): the numerical feature to analyse
        
        Returns: histogram plot with KDE
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def analyse_all(self, data: pd.DataFrame):
        """Histogramme + KDE pour toutes les variables numériques
        
        Parameters:
        data (DataFrame): the dataset
        
        Returns: grid of histogram plots with KDE
        """
        numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        n_cols = 3
        n_rows = len(numerical_cols) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten()
        
        for i, feature in enumerate(numerical_cols):
            sns.histplot(data[feature], kde=True, bins=30, ax=axes[i])
            axes[i].set_title(f"Distribution of {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")
        
        # Supprimer les subplots vides
        for j in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    
    def analyse(self, data: pd.DataFrame, feature: str):
        """Countplot pour une seule variable catégorielle
        
        Parameters:
        data (DataFrame): the dataset
        feature (str): the categorical feature to analyse
        
        Returns: count plot
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=data, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def analyse_all(self, data: pd.DataFrame):
        """Countplot pour toutes les variables catégorielles
        
        Parameters:
        data (DataFrame): the dataset
        
        Returns: grid of count plots
        """
        categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
        
        n_cols = 3
        n_rows = len(categorical_cols) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten()
        
        for i, feature in enumerate(categorical_cols):
            sns.countplot(x=feature, data=data, palette="muted", ax=axes[i])
            axes[i].set_title(f"Distribution of {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis="x", rotation=45)
        
        # Supprimer les subplots vides
        for j in range(len(categorical_cols), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str = None):
        """
        Analyse une seule feature si feature est fourni, 
        toutes les features sinon.

        Parameters:
        df (pd.DataFrame): the dataset
        feature (str): optional - the feature to analyse
        """
        if feature:
            self._strategy.analyse(df, feature)
        else:
            self._strategy.analyse_all(df)