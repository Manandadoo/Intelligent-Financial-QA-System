# Intelligent-Financial-QA-System

This project uses Natural Language Processing (NLP) techniques to build an intelligent question-answering system that extracts, analyzes, and interprets financial content from Reddit and Yahoo Finance. It identifies key sentiment, topics, and contextual information related to financial entities such as companies or stocks.

## 📌 Features

- Scrapes Reddit posts using PRAW.
- Retrieves Yahoo Finance news articles for a given ticker.
- Cleans and preprocesses financial text.
- Performs sentiment analysis using HuggingFace transformers.
- Topic modeling with LDA.
- Contextual similarity using sentence embeddings.
- Named Entity Recognition (NER) with spaCy.

## 📁 Project Structure

```
Intelligent-Financial-QA-System/
├── reddit_data.py                     # Fetch Reddit posts about finance topics
├── check_yahoo_3.py                  # Extract financial news from Yahoo Finance
├── cleaned_data_and_key_context.py   # Text preprocessing, topic modeling, sentiment/context extraction
├── final3.py                         # Final NLP pipeline combining various models
└── processed_articles_with_topics_and_products.csv (expected) 
```

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Intelligent-Financial-QA-System.git
   cd Intelligent-Financial-QA-System
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following models are available:
   - `en_core_web_sm` from spaCy
   - `sentence-transformers` models
   - HuggingFace transformers for sentiment analysis

   You can install the spaCy model using:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🧠 Technologies Used

- Python
- PRAW (Reddit API)
- BeautifulSoup (Yahoo scraping)
- spaCy (NER)
- Transformers (sentiment analysis)
- SentenceTransformers (semantic similarity)
- Scikit-learn (TF-IDF, LDA)
- Pandas, NumPy

## 🧪 How to Run

1. Run `reddit_data.py` to fetch Reddit posts.
2. Run `check_yahoo_3.py` to fetch Yahoo Finance news.
3. Run `cleaned_data_and_key_context.py` to clean and analyze the data.
4. Run `final3.py` to apply advanced NLP tasks like NER and similarity matching.

## 📌 Notes

- Reddit API credentials are hardcoded — you may want to replace them with your own from [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps).
- Ensure you have the CSV file `processed_articles_with_topics_and_products.csv` in the root directory for `final3.py`.
