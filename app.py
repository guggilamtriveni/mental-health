import streamlit as st
import pickle
import re

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Streamlit UI
st.set_page_config(page_title="Mental Health NLP App", layout="centered")

st.title("🧠 Mental Health Text Classification")
st.write("Enter text below to predict mental health category")

user_input = st.text_area("Enter your text here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        clean = clean_text(user_input)
        vector = vectorizer.transform([clean])
        prediction = model.predict(vector)[0]
        st.success(f"Predicted Category: **{prediction}**")
