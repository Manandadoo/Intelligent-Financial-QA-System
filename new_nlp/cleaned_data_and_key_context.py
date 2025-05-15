'sentiment_label', 'sentiment_score', 'context'

import pandas as pd
import spacy
import re
import string
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


# Load spaCy model for NER (Named Entity Recognition)
nlp = spacy.load("en_core_web_sm")

# Load stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load the CSV file containing articles
df = pd.read_csv("company_news_full.csv")

# Remove articles with content length less than 80 words
df = df[df['Content'].astype(str).apply(lambda x: len(x.split()) >= 80)]

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatize and remove stopwords
    return text

# Function to extract key context (using NER and important keywords)
def extract_context(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT", "MONEY", "PERCENT", "DATE"]]
    nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    context = set(entities + nouns)
    return ", ".join(context)

# Apply preprocessing and context extraction
df["cleaned_content"] = df["Content"].astype(str).apply(clean_text)
df["context"] = df["cleaned_content"].apply(extract_context)
df["cleaned_data"] = df["cleaned_content"]

# TF-IDF Vectorization for topic modeling
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_content"])

# Apply LDA for topic modeling
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_matrix = lda.fit_transform(tfidf_matrix)

# Display the top words for each topic
topics = []
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in top_words_idx]
    topics.append(", ".join(top_words))

df["topic"] = [topics[i % len(topics)] for i in range(len(df))]

# Sentence Embedding using Sentence-BERT
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = sentence_model.encode(df["cleaned_content"].tolist())

df["sentence_embeddings"] = list(sentence_embeddings)

# # Function to identify articles talking about a specific product
def extract_products(text, products=["iphone", "macbook", "apple"]):
    doc = nlp(text)
    product_mentions = [ent.text.lower() for ent in doc.ents if ent.label_ == "PRODUCT"]
    mentioned_products = [product for product in products if product in product_mentions]
    return ", ".join(mentioned_products)

df["mentioned_products"] = df["cleaned_content"].apply(lambda x: extract_products(x))

# Load sentiment analysis and summarization pipelines
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", clean_up_tokenization_spaces=True)

# Initialize the tokenizer for tokenization length control
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Function to chunk text into smaller parts with respect to the max token length (512 tokens)
def chunk_text(text, max_length=512):
    # Tokenize text into tokens
    tokens = tokenizer.tokenize(text)
    
    # Chunk the text by the max_length constraint
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokenizer.convert_tokens_to_string(tokens[i:i + max_length])
        chunks.append(chunk)
    return chunks

# Function to get sentiment and Key_context for each article
def extract_sentiment_and_context(row):
    content = row['cleaned_content']
    
    # Sentiment Analysis (handling long texts by chunking)
    sentiment = sentiment_analyzer(content[:512])[0]  # Limit the content to 512 tokens
    
    # Summarization (Context Extraction)
    chunks = chunk_text(content)  # chunk the content if it's too long
    summarized_chunks = [summarizer(chunk)[0]['summary_text'] for chunk in chunks]
    Key_context = ' '.join(summarized_chunks)  # Combine all chunk summaries
    
    return sentiment['label'], sentiment['score'], Key_context
# Save the processed data to CSV
df[['sentiment_label', 'sentiment_score', 'Key_context']] = df.apply(extract_sentiment_and_context, axis=1, result_type="expand")

df[["Title", "Ticker", "Date", "Content", "context", "topic", "cleaned_data", 'sentiment_label', 'sentiment_score', 'Key_context']].to_csv("Cleaned_KeyContext_sent_keyWords.csv", index=False)

print("✅ Data processing complete. Saved processed articles with topics, context, cleaned data, and product mentions to 'processed_articles_with_topics_and_products.csv'.")