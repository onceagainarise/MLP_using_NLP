import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load language detected data
data = pd.read_csv('C:/Users/nishi/Documents/project/Code/data/language_detection_data.csv')

# Display available columns
print("Available columns:", data.columns)

# Check for missing values
if data['cleaned_text'].isnull().any() or data['detected_language'].isnull().any():
    print("Missing values found. Dropping rows with NaN values...")
    data = data.dropna(subset=['cleaned_text', 'detected_language'])

# Verify that there are no more missing values
if data['cleaned_text'].isnull().any() or data['detected_language'].isnull().any():
    print("There are still missing values in the data.")
else:
    print("No missing values found. Proceeding with classification.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['detected_language'], test_size=0.2)

    # Convert text data into features using CountVectorizer
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    # Predict and evaluate the model
    y_pred = clf.predict(X_test_vec)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save classifier model (optional)
    import joblib
    joblib.dump(clf, 'C:/Users/nishi/Documents/project/Code/model/language_classifier_model.pkl')
    joblib.dump(vectorizer, 'C:/Users/nishi/Documents/project/Code/model/language_vectorizer.pkl')
