from pathlib import Path 


try : 
    ROOT_DIR=Path(__file__).parent.parent
except NameError:
    ROOT_DIR=Path().resolve().parent
 
 
# Main Folder
CONFIG_DIR=ROOT_DIR/'config'   
DATA_DIR=ROOT_DIR/"data"
EDA_DIR=ROOT_DIR/"Eda_analysis"



#files
ZIP_FILE=DATA_DIR/'archive.zip'
data__raw_dir=EDA_DIR/'Telco_customer_churn.csv'



    