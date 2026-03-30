import pandas as pd
df = pd.read_csv('~/autodl-tmp/ptbxl_data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv')
print(df.head())
print(df.diagnostic_superclass.value_counts())