import pandas as pd
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------- Text Cleaning --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join(w for w in text.split() if w not in stop_words)
    return text

# -------- Load Data --------
train_data = pd.read_csv(
    "dataset/train_data.txt",
    sep=":::",
    engine="python",
    names=["id", "title", "plot_summary", "genre"]
)

train_data["clean_plot"] = train_data["plot_summary"].apply(clean_text)

# -------- Vectorization --------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train_data["clean_plot"])
y = train_data["genre"]

# -------- Train Model --------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -------- Streamlit UI --------
st.title("🎬 Movie Genre Prediction App")

plot_input = st.text_area("Enter movie plot summary:")

if st.button("Predict Genre"):
    if plot_input.strip() == "":
        st.warning("Please enter a plot summary.")
    else:
        cleaned_plot = clean_text(plot_input)
        plot_vector = vectorizer.transform([cleaned_plot])
        prediction = model.predict(plot_vector)[0]
        st.success(f"Predicted Genre: **{prediction}**")
