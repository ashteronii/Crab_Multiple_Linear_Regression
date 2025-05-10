import pandas as pd

# Load the Crab Age Prediction dataset from the specified CSV file.
# This dataset is publicly available on Kaggle: https://www.kaggle.com/datasets/sidhus/crab-age-prediction/data
crabs = pd.read_csv("CrabAgePrediction.csv")

# Cleans and preprocesses the crab dataset.
# Steps:
#    - Drop missing values.
#    - Remove duplicate rows.
#    - Remove invalid height values.
#    - Filter out implausible dimensions.
#    - One-hot encode the 'Sex' column.
# This ensures that we only work with complete and unique records for analysis.
crabs = crabs.dropna().drop_duplicates()
crabs = crabs.query("Height != 0")
crabs = crabs.query("Length > Diameter")
crabs = pd.get_dummies(crabs, columns=['Sex'])

# Save the cleaned version of the dataset to a new CSV file called 'Crab_Cleaned.csv'.
# This cleaned dataset will be used for training and further analysis.
crabs.to_csv("Crab_Cleaned.csv")