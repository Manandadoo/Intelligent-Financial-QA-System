import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def get_company_news(ticker):
    """Fetches news articles related to a company from Yahoo Finance"""
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount=20"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)    #sends a GET request to the Yahoo Finance API
    data = response.json()

    news_data = []

    for article in data.get("news", []):
        title = article.get("title", "No Title")
        yahoo_link = article.get("link", "")
        publisher = article.get("publisher", "Unknown")
        date = article.get("providerPublishTime", "")

        # Convert timestamp to readable date
        if date:
            date = datetime.utcfromtimestamp(date).strftime("%Y-%m-%d %H:%M:%S")

        # Extract the external news link
        external_link = extract_external_link(yahoo_link)

        # Scrape the full article content
        content = scrape_full_article(external_link)

        news_data.append({
            "Ticker": ticker,
            "Title": title,
            "Date": date,
            "Publisher": publisher,
            "Yahoo_URL": yahoo_link,
            "External_URL": external_link,
            "Content": content
        })

    return news_data

def extract_external_link(yahoo_url):
    """Extracts external source link from a Yahoo Finance news article"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(yahoo_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Find external news link inside the article
        link_tag = soup.find("a", {"class": "link caas-button"})
        if link_tag and "href" in link_tag.attrs:
            return link_tag["href"]
        
        return yahoo_url  # If no external link, return Yahoo's URL

    except Exception as e:
        return f"Error extracting link: {str(e)}"

def scrape_full_article(url):
    """Scrapes full article content from the given URL"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract all text from the article body
        article_body = soup.find("article") or soup.find("div", {"class": "body"}) or soup.find("div", {"class": "content"})
        if not article_body:
            return "Content not found."

        # Extract and clean paragraphs
        paragraphs = article_body.find_all("p")
        content = "\n".join([p.get_text().strip() for p in paragraphs])

        return content if content else "Content unavailable."

    except Exception as e:
        return f"Error fetching article: {str(e)}"

# Fetch news for Apple & Samsung
apple_news = get_company_news("AAPL")
samsung_news = get_company_news("Samsung")

# Convert to CSV
df = pd.DataFrame(apple_news + samsung_news)
df.to_csv("company_news_full.csv", index=False)

print("✅ Full news content saved successfully!")
