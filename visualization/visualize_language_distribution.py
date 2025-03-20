import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('C:/Users/language_detection_data.csv')


print("Available columns:", data.columns)


if 'detected_language' in data.columns:
    
    plt.figure(figsize=(12, 12))
    sns.countplot(y=data['detected_language'], order=data['detected_language'].value_counts().index)
    plt.title('Language Distribution')
    plt.xlabel('Number of Tweets')
    plt.ylabel('Detected Language')
    plt.show()
else:
    print("Error: 'detected_language' column is missing in the DataFrame.")
