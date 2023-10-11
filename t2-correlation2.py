import yaml
import yfinance as yf
import re
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd

with open('noticias_by_time.yaml', 'r', encoding='utf-8') as file:
    news_data = yaml.safe_load(file)

def process_company_news(company_name, news_list):
    news_data[company_name] = []
    for news_item in news_list:
        news_title = news_item['Título']
        news_content = news_item['Conteúdo']
        news_date = news_item['Data']
        news_data[company_name].append({
            'Título': news_title,
            'Conteúdo': news_content,
            'Data': news_date
        })

nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)    
    words = text.lower().split()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')) 
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = TextBlob(" ".join(tokens)).words.lemmatize()
    return ' '.join(lemmatizer)

sentiments = {}
stock_data = {}

start_date = '2023-10-05'
end_date = '2023-10-06'

for company, news_list in news_data.items():
    sentiments[company] = []
    stock_data[company] = []
    for news in news_list:
        news_date = news['Data'] 
        if news_date: 
            preprocessed_news = preprocess_text(news['Conteúdo'])
            sentiment = TextBlob(preprocessed_news)
            sentiment_polarity = sentiment.sentiment.polarity
            if sentiment_polarity > 0:
                sentiment = 'Positivo'
            elif sentiment_polarity < 0:
                sentiment = 'Negativo'
            else:
                sentiment = 'Neutro'
            sentiments[company].append({"titulo": news['Título'],"sentiment": sentiment, "sentiment_polarity": sentiment_polarity, "data": news_date})
    
    stock = yf.download(company, start=start_date, end=end_date, interval='1h')
    stock_data[company] = stock['Close'].diff()

sentiment_polarity_by_company = {}
stock_value_by_company = {}

print(stock_data)

for empresa, dados in sentiments.items():
    sentiment_polarity_values = []
    stock_value_values = []
    for dado in dados:
        sentiment_polarity_values.append(dado['sentiment_polarity'])
        
        news_date_str = dado['data']
        news_date = pd.to_datetime(news_date_str, utc=True)  # Converter para Timestamp com UTC
        news_date = news_date.tz_convert('-04:00:00')
        nearest_stock_date = min(stock_data[empresa].index, key=lambda x: abs(x - news_date))

        stock_value = stock_data[empresa].loc[nearest_stock_date]
        stock_value_values.append(stock_value)

    sentiment_polarity_by_company[empresa] = sentiment_polarity_values
    stock_value_by_company[empresa] = stock_value_values

correlation = {}

for empresa, sentiment_values in sentiment_polarity_by_company.items():
    correlation[empresa] = np.corrcoef(sentiment_values, stock_value_by_company[empresa])[0, 1]

    print(f"Empresa: {empresa}")
    print("Sentiment Polarity Values:", sentiment_values)
    print("Stock Value Values:", stock_value_by_company[empresa])
    print("Correlação:", correlation[empresa])

print('Correlação\n', correlation)
