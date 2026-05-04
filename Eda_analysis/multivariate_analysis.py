import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from abc import ABC, abstractmethod


class MultivariateAnalysisTemplate(ABC):
    
    def analyze(self, df: pd.DataFrame, features: list = None, method: str = "spearman"):
        """
        Orchestrate the full multivariate analysis.

        Parameters:
        df (pd.DataFrame): the dataset
        features (list): optional - specific features for the pairplot
        method (str): correlation method — default spearman
        """
        self.generate_correlation_heatmap(df, method)
        self.generate_pairplot(df, features)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame, method: str):
        """
        Generate and display a correlation heatmap.

        Parameters:
        df (pd.DataFrame): the dataset
        method (str): correlation method — pearson, spearman or kendall
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame, features: list):
        """
        Generate and display a pairplot.

        Parameters:
        df (pd.DataFrame): the dataset
        features (list): optional - specific features to plot
        """
        pass


class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):

    def generate_correlation_heatmap(self, df: pd.DataFrame, method: str = "spearman"):
        """
        Generates and displays a correlation heatmap.
        Excludes object columns — Spearman handles both numerical and ordinal.

        Parameters:
        df (pd.DataFrame): the dataset
        method (str): correlation method — default spearman

        Returns:
        None: displays a correlation heatmap
        """
        data = df.select_dtypes(exclude=["object"])

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            data.corr(method=method),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5
        )
        plt.title(f"Correlation Heatmap ({method})")
        plt.tight_layout()
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame, features: list = None):
        """
        Generates and displays a pairplot colored by Satisfaction Score.
        If no features provided, uses all numerical columns.

        Parameters:
        df (pd.DataFrame): the dataset
        features (list): optional - specific features to plot

        Returns:
        None: displays a pairplot
        """
        if features:
            # S'assurer que Satisfaction Score est toujours inclus pour le hue
            if "Satisfaction Score" not in features:
                features = features + ["Satisfaction Score"]
            data = df[features]
        else:
            data = df.select_dtypes(exclude=["object"])

        sns.pairplot(
            data,
            hue="Satisfaction Score",
            diag_kind="kde",
            plot_kws={"alpha": 0.5}
        )
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.tight_layout()
        plt.show()
