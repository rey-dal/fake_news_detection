from flask import Flask, render_template, request
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load pre-trained model and vectorizer
classifier = pickle.load(open('model.pkl', 'rb'))
CV = pickle.load(open('count_vectorizer.pkl', 'rb'))

def preprocess_text(sample_news):
    # Text preprocessing function
    sample_news = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_news)
    sample_news = sample_news.lower()
    sample_news_words = sample_news.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    sample_news_words = [word for word in sample_news_words if word not in stop_words]
    
    # Stemming
    ps = PorterStemmer()
    final_news = [ps.stem(word) for word in sample_news_words]
    final_news = ' '.join(final_news)
    
    return final_news

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    processed_news = preprocess_text(news_text)
    vectorized_news = CV.transform([processed_news]).toarray()
    prediction = classifier.predict(vectorized_news)[0]
    
    result = "Fake News" if prediction == 1 else "Real News"
    return render_template('index.html', prediction_text=f'This is: {result}')

if __name__ == '__main__':
    app.run(debug=True)
