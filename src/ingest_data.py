import os 
import zipfile
from abc import ABC, abstractmethod
import pandas as pd 
import sys
from pathlib import Path
import importlib

from config.ingestor_config import (
    SUPPORTED_EXTENSIONS, 
    TELCO_TABLE_FILES,
    MERGE_KEY_CUSTOMER,
    MERGE_KEY_LOCATION
)
import config.ingestor_config
importlib.reload(config.ingestor_config)

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str, filename: str = None) -> pd.DataFrame:
        "abstract method to ingest data for a given type of file"
        pass


class ZipDataIngestor(DataIngestor):
    'this class is for the extraction of csv and excel files in a zip file'

    def ingest(self, file_path: str, filename: str = None) -> pd.DataFrame:
        """
        Extract and load files from a zip.

        Parameters:
            file_path (str): path to the zip file
            filename (str): optional - specific file to load

        Returns:
            pd.DataFrame: loaded or merged dataframe
        """
        if not str(file_path).endswith('.zip'):
            raise ValueError('the file provided is not a zip')

        with zipfile.ZipFile(file_path, 'r') as f:
            f.extractall('extracted_data')

        extracted_files = os.listdir('extracted_data')
        supported_files = [f for f in extracted_files if f.endswith(SUPPORTED_EXTENSIONS)]

        if len(supported_files) == 0:
            raise ValueError(f'No supported files found. Extensions: {SUPPORTED_EXTENSIONS}')

        # Cas 1 — fichier spécifique demandé
        if filename:
            if filename not in supported_files:
                raise FileNotFoundError(
                    f"{filename} not found. Available: {supported_files}"
                )
            file_path = os.path.join("extracted_data", filename)
            return pd.read_csv(file_path) if filename.endswith('.csv') else pd.read_excel(file_path)

        # Cas 2 — un seul fichier
        if len(supported_files) == 1:
            file_path = os.path.join("extracted_data", supported_files[0])
            return pd.read_csv(file_path) if supported_files[0].endswith('.csv') else pd.read_excel(file_path)

        # Cas 3 — plusieurs fichiers IBM Telco → merge automatique
        telco_files = [f for f in supported_files if f in TELCO_TABLE_FILES.values()]
        if len(telco_files) == len(TELCO_TABLE_FILES):
            return self._merge_telco_files()

        raise ValueError(
            f"Multiple files found: {supported_files}. "
            "Please specify a filename."
        )
        
    #uniquement pour le telco pour ce dataset (à supprimer plus tard)
    def _merge_telco_files(self) -> pd.DataFrame:
        dataframes = {
        name: pd.read_excel(os.path.join("extracted_data", filename))
        for name, filename in TELCO_TABLE_FILES.items()
    }

    # Base — demographics
        df = dataframes["demographics"]

        for name in ["location", "services", "status"]:
          other = dataframes[name]
          new_cols = [col for col in other.columns if col not in df.columns or col == "Customer ID"]
          df = df.merge(other[new_cols], on="Customer ID", how="left")

          # Population — clé différente
        population = dataframes["population"]
        new_cols = [col for col in population.columns if col not in df.columns or col == "Zip Code"]
        df = df.merge(population[new_cols], on="Zip Code", how="left")

        return df
     
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")