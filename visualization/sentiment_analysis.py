import pandas as pd
from textblob import TextBlob

# Load preprocessed data
data = pd.read_csv('C:/Users/nishi/Documents/project/Code/data/preprocessed_data.csv')

if data['cleaned_text'].isnull().any():
    print("Missing values found in cleaned_text. Filling with empty strings.")
    data['cleaned_text'].fillna('', inplace=True)  # Fill NaNs with an empty string


# Define a function to get sentiment polarity for a text
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Apply sentiment analysis to the cleaned text
data['sentiment'] = data['cleaned_text'].apply(get_sentiment)

# Display the sentiment scores
print(data[['cleaned_text', 'sentiment']].head())

# Save data with sentiment analysis
data.to_csv('C:/Users/nishi/Documents/project/Code/data/sentiment_data.csv', index=False)