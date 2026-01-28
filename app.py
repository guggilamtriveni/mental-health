import streamlit as st
import pickle
import numpy as np
#from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved files
model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Parameters (same as training)
max_len = 100

# UI
st.title("NLP Text Classification using LSTM")
st.write("Enter text to predict the class")

user_input = st.text_area("Enter your text here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        pad = pad_sequences(seq, maxlen=max_len, padding='post')
        pred = model.predict(pad)
        label = le.inverse_transform([np.argmax(pred)])
        st.success(f"Prediction: {label[0]}")
