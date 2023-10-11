import yaml
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from expand import expand_contractions
import string
from nltk.tokenize import word_tokenize
from graphic import plot_price_sentiment_relationship

with open('noticias.yaml', 'r', encoding='utf-8') as file:
    news_data = yaml.safe_load(file)

# Função para processar as notícias de uma empresa
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

# Pré-processamento de texto e análise de sentimentos
nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)    
    words = text.lower().split()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))  # Você pode alterar o idioma conforme necessário
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = TextBlob(" ".join(tokens)).words.lemmatize()
    return ' '.join(lemmatizer)

sentiments = {}
stock_data = {}

# Iterar sobre as empresas e suas notícias
for company, news_list in news_data.items():
    sentiments[company] = []
    stock_data[company] = []
    for news in news_list:
        preprocessed_news = preprocess_text(news['Conteúdo'])
        sentiment = TextBlob(preprocessed_news)
        sentiment_polarity = sentiment.sentiment.polarity
        if sentiment_polarity > 0:
            sentiment = 'Positivo'
        elif sentiment_polarity < 0:
            sentiment = 'Negativo'
        else:
            sentiment = 'Neutro'
        sentiments[company].append({"titulo": news['Título'],"sentiment": sentiment, "sentiment_polarity": sentiment_polarity, "data": news['Data']})
    
    stock = yf.download(company, start='2023-01-01', end='2023-10-10')
    stock_data[company] = stock['Close'].diff()


# Agora, news_data contém as notícias carregadas a partir do arquivo YAML
# print('Sentiment\n', sentiments)
# print('Stock\n', stock_data)

combined_data2 = {}
for empresa, sentimentos in sentiments.items():
    combined_data2[empresa] = []
    for sentimento in sentimentos:
        data = sentimento['data']
        stock_value = stock_data[empresa].get(data)
        if stock_value is not None:
            sentimento['stock_value'] = stock_value
            combined_data2[empresa].append(sentimento)

print(combined_data2['AAPL'])

# Arrays separados para sentimentos e stock data por empresa
sentiment_polarity_by_company = {}
stock_value_by_company = {}

# Relacionar os arrays por empresa
for empresa, dados in combined_data2.items():
    sentiment_polarity_values = []
    stock_value_values = []
    for dado in dados:
        sentiment_polarity_values.append(dado['sentiment_polarity'])
        stock_value_values.append(dado['stock_value'])

    sentiment_polarity_by_company[empresa] = sentiment_polarity_values
    stock_value_by_company[empresa] = stock_value_values


correlation = {}

# Resultados por empresa
for empresa, sentiment_values in sentiment_polarity_by_company.items():
    print(sentiment_values, stock_value_by_company)
    correlation[empresa] = np.corrcoef(sentiment_values, stock_value_by_company[empresa])[0,1]

    print(f"Empresa: {empresa}")
    print("Sentiment Polarity Values:", sentiment_values)
    print("Stock Value Values:", stock_value_by_company[empresa])

# Suponha que você já tenha as informações em sentiments e stocks_data
print('Correlação\n', correlation)

