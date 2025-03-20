import pandas as pd
import langid

try:
    data = pd.read_csv('C:/Users/preprocessed_data.csv')

 
    print("Available columns:", data.columns)

    
    data['cleaned_text'] = data['cleaned_text'].astype(str)

   
    data['detected_language'] = data['cleaned_text'].apply(lambda x: langid.classify(x)[0])

    
    print("Detected languages:")
    print(data[['cleaned_text', 'detected_language']].head())


    data.to_csv('C:/Users/nishi/Documents/project/Code/data/language_detection_data.csv', index=False)

except Exception as e:
    print("An error occurred:", e)




