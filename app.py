from flask import Flask, render_template, request
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load model and vectorizer
with open('model/fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    news_transformed = vectorizer.transform([news])
    prediction = model.predict(news_transformed)
    result = "Real News ✅" if prediction[0] == 1 else "Fake News ❌"
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
