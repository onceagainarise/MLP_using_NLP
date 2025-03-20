import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('C:/Users/nishi/Documents/project/Code/data/language_detection_data.csv')

# Print available columns
print("Available columns:", data.columns)

# Check if 'detected_language' is in the DataFrame
if 'detected_language' in data.columns:
    # Visualize the distribution of detected languages
    plt.figure(figsize=(12, 12))
    sns.countplot(y=data['detected_language'], order=data['detected_language'].value_counts().index)
    plt.title('Language Distribution')
    plt.xlabel('Number of Tweets')
    plt.ylabel('Detected Language')
    plt.show()
else:
    print("Error: 'detected_language' column is missing in the DataFrame.")
