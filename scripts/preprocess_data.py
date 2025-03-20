import pandas as pd
import re


def clean_text(text):
    if isinstance(text, str):
      
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'[^A-Za-z\s]', '', text)
       
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''


try:
    
    data = pd.read_csv('C:/Userscleaned_data.csv', delimiter='\t')
    
   
    print("Initial columns:", data.columns)

   
    if 'Tweet' in data.columns:
        
        data['cleaned_text'] = data['Tweet'].apply(clean_text)

        
        print("Data after cleaning:")
        print(data[['Tweet', 'cleaned_text']].head())  

       
        data.to_csv('C:/Users/preprocessed_data.csv', index=False)
    else:
        print("Column 'Tweet' not found in the input data.")

except pd.errors.ParserError as e:
    print("Error parsing the CSV file:", e)
except Exception as e:
    print("An error occurred:", e)
