#Email Spam detection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import string
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Display first 5 rows
print(df.head())

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print(df.head())

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print(df['label'].value_counts())

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    
    return ' '.join(words)

df['message'] = df['message'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

def predict_email(text):
    text = preprocess_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    
    if prediction[0] == 1:
        return "🚨 Spam Email"
    else:
        return "✅ Not Spam"

# Test Example
print(predict_email("Congratulations! You have won a free gift card"))
print(predict_email("Hey, are we meeting tomorrow?"))
