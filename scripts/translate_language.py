import pandas as pd
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
from huggingface_hub import login


login(token="hf_XXXXXXXXXXXXXXXXXXXXXXXXX", add_to_git_credential=True)


data = pd.read_csv('C:/Users/language_detection_data.csv')


print("Available columns:", data.columns)


print(data.head())


model_name = "Helsinki-NLP/opus-mt-mul-en"  

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translate_tweet(cleaned_text, detected_language):
    if cleaned_text and detected_language != 'en':  
        try:
           
            print(f"Original Text: {cleaned_text}")
            inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True)
            
            
            outputs = model.generate(**inputs)
            
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            
            print(f"Translated Text: {translated_text}")
            
            return translated_text
        except Exception as e:
            print(f"Error translating tweet: {cleaned_text}, Error: {e}")
            return cleaned_text 
    else:
        return cleaned_text 

print("Language check for first 5 rows:")
for i in range(5):
    print(f"Text: {data.iloc[i]['cleaned_text']}, Detected Language: {data.iloc[i]['detected_language']}")

subset_data = data.iloc[:20].copy() 
try:
    subset_data['translated_tweet'] = subset_data.apply(
        lambda row: translate_tweet(row['cleaned_text'], row['detected_language']), axis=1
    )
except Exception as e:
    print(f"Error occurred: {e}")


data.to_csv('C:/Users/nishi/Documents/project/Code/data/translated_data.csv', index=False)

print("Translation complete. Saved to 'translated_data.csv'.")
