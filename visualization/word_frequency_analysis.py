import pandas as pd
from collections import Counter


data = pd.read_csv('C:/Users.csv')

def word_freq_by_language(language, data):
 
    lang_data = data[data['detected_language'] == language]['cleaned_text']
    
    lang_data = lang_data.fillna('').astype(str) 
    

    words = ' '.join(lang_data).split()
    
   
    word_counts = Counter(words)
    
    return word_counts


english_freq = word_freq_by_language('en', data)

print(english_freq.most_common(10)) 

