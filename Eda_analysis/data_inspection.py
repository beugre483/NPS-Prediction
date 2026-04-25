import pandas as pd
import os 
import sys 
from abc import ABC, abstractmethod
import seaborn as sns 
import matplotlib.pyplot as plt 



#Data inspection with de Strategy design pattern 


class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect_data(self,data:pd.DataFrame):
        """perform a type of DataInspection
        
        parameters : 
        data (pd.Dataframe) : the dataset for the inspection
        
        Return (None) : Depend on the type of inspection 
    """
        pass




class DataTypeInspection(DataInspectionStrategy):
    def inspect_data(self,data:pd.DataFrame):
        """_check the type of all the variables in the dataset_

        parameters:
            data (pd.DataFrame): _The dataset of analsyse_
            
        return: 
           The type of the variables
        """
        
        assert isinstance(data,pd.DataFrame),'veuillez donner une variable pandas'
        
        print("\n Types of variables of the Dataset :")
        print(data.info())






class DataSummaryStaticticsInspection(DataInspectionStrategy):
    def inspect_data(self,data:pd.DataFrame):
        """perform statistics inspection in the dataset for both 
        numerical and categorical features 

        parameters:
            data (pd.DataFrame): the dataset of the analyse
            
        return: 
        descriptive statistics
    
        """
        #verifiy the type of variables give in the method 
        assert isinstance(data,pd.DataFrame),"Veuillez donner une variable pandas"
        print("\n Summary Statitics for numerical features :")
        print(data.describe().to_string())
        print("\n Summary Statistics for categorical features :")
        print(data.describe(include=['O']).to_string())       
        
        
        
        
class DataMissingValuesInspection(DataInspectionStrategy):
    def  inspect_data(self,data:pd.DataFrame):
        """ perform analyse of missing values 

        parameters:
            data (pd.Dataframe): _dataset of analyse_
            
        return : a heatmap that highligth column with missing values
        """
        assert isinstance(data, pd.DataFrame),"veuillez donner une variable pandas "
        
        plt.figure(figsize=(12,6))
        sns.heatmap(data.isnull(),cbar=False, cmap='viridis')
        plt.title('Hheatmap des valeurs manquantes')
        plt.show()
        




# construct the context class

class DataInspection(): 
    def __init__(self,strategy : DataInspectionStrategy):
        
        """ 
        Initialise the DataInspectionStrategy with a specifiq inspection strategy
        Strategy(DataInspection): the strategy to be used for the inspection 
        """
        self.strategy=strategy
        
    def set_strategy(self,strategy: DataInspectionStrategy) :
        """set a new strategy 
        """
        self.strategy=strategy
        
    def do_inspection(self,data:pd.DataFrame):
        
        """
        aplly te strategy of inspection
        """
        self.strategy.inspect_data(data)