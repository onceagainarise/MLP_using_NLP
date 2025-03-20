import streamlit as st
import langid
from transformers import pipeline
import joblib
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


model = joblib.load("model/language_classifier_model.pkl")
vectorizer = joblib.load("model/language_vectorizer.pkl")


st.title("Multilingual Language Detection and Translation System")


text = st.text_area("Enter text here:", height=150)


option = st.selectbox(
    "Choose an action:",
    ["Detect Language", "Translate Text", "Visualize Word Cloud"]
)


if st.button("Run"):
    try:
        if option == "Detect Language":
            
            if text.strip():
                lang_detect = langid.classify(text)[0]
                st.write(f"Detected Language: {lang_detect}")
            else:
                st.warning("Please enter some text to detect the language.")

        elif option == "Translate Text":
            
            if text.strip():
                translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
                translation = translator(text, max_length=400)
                st.write(f"Translation: {translation[0]['translation_text']}")
            else:
                st.warning("Please enter some text to translate.")

        elif option == "Visualize Word Cloud":
            
            if text.strip():
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    stopwords=STOPWORDS
                ).generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.warning("Please enter some text to generate a word cloud.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
