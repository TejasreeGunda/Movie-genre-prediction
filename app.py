import pandas as pd

train_data = pd.read_csv(
    "dataset/train_data.txt",
    sep=":::",
    engine="python",
    names=["id", "title", "plot_summary", "genre"]
)

print(train_data.head())


import pandas as pd

test_data = pd.read_csv(
    "dataset/test_data.txt",
    sep=":::",
    engine="python",
    names=["id", "title", "plot_summary"]
)

print(test_data.head())

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join(w for w in text.split() if w not in stop_words)
    return text

train_data['clean_plot'] = train_data['plot_summary'].apply(clean_text)
test_data['clean_plot'] = test_data['plot_summary'].apply(clean_text)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_train = vectorizer.fit_transform(train_data['clean_plot'])
y_train = train_data['genre']

X_test = vectorizer.transform(test_data['clean_plot'])

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

model.fit(X_tr, y_tr)
pred = model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, pred))

test_data['predicted_genre'] = model.predict(X_test)

print(test_data[['title', 'predicted_genre']].head())


test_data.to_csv("predicted_genres.csv", index=False)

