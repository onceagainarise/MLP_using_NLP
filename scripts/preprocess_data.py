import pandas as pd
import re

# Function to clean the text
def clean_text(text):
    if isinstance(text, str):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters and numbers
        text = re.sub(r'[^A-Za-z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''

# Load the data
try:
    # Change delimiter to tab (\t) if your file is tab-delimited
    data = pd.read_csv('C:/Users/nishi/Documents/project/Code/data/cleaned_data.csv', delimiter='\t')
    
    # Print available columns for debugging
    print("Initial columns:", data.columns)

    # Check if 'Tweet' column exists
    if 'Tweet' in data.columns:
        # Apply the cleaning function to the 'Tweet' column
        data['cleaned_text'] = data['Tweet'].apply(clean_text)

        # Print to check the DataFrame
        print("Data after cleaning:")
        print(data[['Tweet', 'cleaned_text']].head())  # Display the relevant columns

        # Save the cleaned data
        data.to_csv('C:/Users/nishi/Documents/project/Code/data/preprocessed_data.csv', index=False)
    else:
        print("Column 'Tweet' not found in the input data.")

except pd.errors.ParserError as e:
    print("Error parsing the CSV file:", e)
except Exception as e:
    print("An error occurred:", e)
