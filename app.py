
from fastapi import FastAPI
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI()

model = joblib.load("mental_health_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

@app.get("/")
def home():
    return {"message": "Mental Health NLP API"}

@app.post("/predict")
def predict(text: str):
    clean = clean_text(text)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    return {"prediction": pred}
