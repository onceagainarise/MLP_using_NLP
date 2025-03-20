import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load language detected data
data = pd.read_csv('C:/Users/nishi/Documents/project/Code/data/language_detection_data.csv')

def generate_wordcloud(language, data):
    # Filter the DataFrame for the specified language
    lang_data = data[data['detected_language'] == language]['cleaned_text']
    
    # Check the filtered data
    print("Filtered Data:")
    print(lang_data)  # Check if there's any data available

    # Ensure all entries are strings and handle NaN values
    lang_data = lang_data.fillna('').astype(str)  # Replace NaNs with empty strings and convert to str
    
    # Join all text data into a single string
    text = ' '.join(lang_data)
    
    # Generate word cloud
    if not text.strip():  # Check if text is empty
        print("No text data available for the specified language.")
    else:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        print("Word cloud generated.")  # Confirm generation

        # Plotting the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Hide axes
        plt.show()
        print("Displaying the word cloud.")  # Confirm display

# Example usage
generate_wordcloud('en', data)