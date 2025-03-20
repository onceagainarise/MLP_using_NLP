import pandas as pd
import langid

try:
    data = pd.read_csv('C:/Users/nishi/Documents/project/Code/data/preprocessed_data.csv')

    # Print available columns for debugging
    print("Available columns:", data.columns)

    # Ensure all values in 'cleaned_text' are strings
    data['cleaned_text'] = data['cleaned_text'].astype(str)

    # Classify the language of the cleaned text
    data['detected_language'] = data['cleaned_text'].apply(lambda x: langid.classify(x)[0])

    # Print to check the DataFrame
    print("Detected languages:")
    print(data[['cleaned_text', 'detected_language']].head())

    # Save the results
    data.to_csv('C:/Users/nishi/Documents/project/Code/data/language_detection_data.csv', index=False)

except Exception as e:
    print("An error occurred:", e)




