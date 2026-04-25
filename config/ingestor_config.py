# config/ingestor_config.py

# Extensions supportées par ZipDataIngestor
SUPPORTED_EXTENSIONS = ('.csv', '.xlsx', '.xls')

# Les 5 fichiers IBM Telco à merger
TELCO_TABLE_FILES = {
    "demographics" : "Telco_customer_churn_demographics.xlsx",
    "location"     : "Telco_customer_churn_location.xlsx",
    "population"   : "Telco_customer_churn_population.xlsx",
    "services"     : "Telco_customer_churn_services.xlsx",
    "status"       : "Telco_customer_churn_status.xlsx",
}

# Clés de merge
MERGE_KEY_CUSTOMER = "Customer ID"   
MERGE_KEY_LOCATION = "Zip Code"
MERGE_KEY_POPULATION = "ID"          