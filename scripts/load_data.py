import pandas as pd


data = pd.read_csv('C:/Users/all_annotated.tsv',sep='\t')


print(data.info())
print(data.head())


data.to_csv('C:/Users/cleaned_data.csv', sep='\t', index= False)
