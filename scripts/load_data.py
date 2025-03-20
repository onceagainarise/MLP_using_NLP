import pandas as pd

# Load the dataset using pandas
data = pd.read_csv('C:/Users/nishi/Documents/project/Code/data/all_annotated.tsv',sep='\t')

# Display basic info
print(data.info())
print(data.head())

# Save cleaned data (optional)
data.to_csv('C:/Users/nishi/Documents/project/Code/data/cleaned_data.csv', sep='\t', index= False)
