import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


data = pd.read_csv('C:/Users/language_detection_data.csv')

def generate_wordcloud(language, data):

    lang_data = data[data['detected_language'] == language]['cleaned_text']
    

    print("Filtered Data:")
    print(lang_data) 
    
    lang_data = lang_data.fillna('').astype(str)  
    
    text = ' '.join(lang_data)
    
    
    if not text.strip(): 
        print("No text data available for the specified language.")
    else:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        print("Word cloud generated.")  

      
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off') 
        plt.show()
        print("Displaying the word cloud.")  


generate_wordcloud('en', data)
