import pandas as pd
from collections import Counter

# Load language detected data
data = pd.read_csv('C:/Users/nishi/Documents/project/Code/data/language_detection_data.csv')

def word_freq_by_language(language, data):
    # Filter the DataFrame for the specified language
    lang_data = data[data['detected_language'] == language]['cleaned_text']
    
    # Ensure all entries are strings and handle NaN values
    lang_data = lang_data.fillna('').astype(str)  # Replace NaNs with empty strings and convert to str
    
    # Join all text data and split into words
    words = ' '.join(lang_data).split()
    
    # Count word frequencies
    word_counts = Counter(words)
    
    return word_counts

# Example usage
english_freq = word_freq_by_language('en', data)

print(english_freq.most_common(10)) 
#print(english_freq)
