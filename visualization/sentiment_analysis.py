import pandas as pd
from textblob import TextBlob


data = pd.read_csv('C:/Users/nishi/Documents/project/Code/data/preprocessed_data.csv')

if data['cleaned_text'].isnull().any():
    print("Missing values found in cleaned_text. Filling with empty strings.")
    data['cleaned_text'].fillna('', inplace=True) 



def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


data['sentiment'] = data['cleaned_text'].apply(get_sentiment)


print(data[['cleaned_text', 'sentiment']].head())


data.to_csv('C:/Users/nishi/Documents/project/Code/data/sentiment_data.csv', index=False)
