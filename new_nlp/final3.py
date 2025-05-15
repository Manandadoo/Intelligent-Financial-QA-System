import pandas as pd
import spacy
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import numpy as np

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load the CSV file containing processed articles
df = pd.read_csv("processed_articles_with_topics_and_products.csv")

# Sentence Embedding Model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Flan-T5 model (using GPU if available)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", device=0)

# Function to find the top 3 relevant articles using the appropriate column for cleaned text
def find_relevant_articles(question):
    question_embedding = sentence_model.encode(question)
    # Use "cleaned_data" if exists, otherwise use "cleaned_content", or fallback to "Content"
    if "cleaned_data" in df.columns:
        contexts = df["cleaned_data"].astype(str).tolist()
    elif "cleaned_content" in df.columns:
        contexts = df["cleaned_content"].astype(str).tolist()
    else:
        contexts = df["Content"].astype(str).tolist()
    
    context_embeddings = sentence_model.encode(contexts)
    similarities = util.cos_sim(question_embedding, context_embeddings)[0]
    
    if similarities is not None and len(similarities) > 0:
        sim_array = similarities.cpu().numpy() if hasattr(similarities, "cpu") else np.array(similarities)
        top_indices = np.argsort(sim_array)[-min(len(sim_array), 3):][::-1]
    else:
        top_indices = []
    
    best_articles = [contexts[idx] for idx in top_indices]
    return best_articles

# Function to find the most relevant part from each article by extracting top 5 sentences
def extract_relevant_part(article, question):
    sentences = article.split('. ')
    sentence_embeddings = sentence_model.encode(sentences)
    question_embedding = sentence_model.encode(question)
    similarities = util.cos_sim(question_embedding, sentence_embeddings)[0]
    sim_array = similarities.cpu().numpy() if hasattr(similarities, "cpu") else np.array(similarities)
    # Get top 5 sentences from the article (if available)
    top_sentence_indices = np.argsort(sim_array)[-min(len(sim_array), 5):][::-1]
    relevant_sentences = [sentences[i] for i in top_sentence_indices]
    return " ".join(relevant_sentences)

# Chunk the content to fit within the model's token limit
def chunk_text(text, max_length=400):
    words = text.split()
    return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

# Function to generate an answer using Flan-T5 with longer output settings
def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"
    # Setting max_new_tokens to 200 to allow for a more detailed answer
    answer = qa_pipeline(input_text, max_new_tokens=200)[0]["generated_text"]
    return answer

# Main function to answer user questions
def answer_question(question):
    relevant_articles = find_relevant_articles(question)
    if relevant_articles:
        # For each relevant article, extract the most relevant parts (top 5 sentences)
        combined_context = " ".join([extract_relevant_part(article, question) for article in relevant_articles])
        chunks = chunk_text(combined_context)
        best_answer = ""
        for chunk in chunks:
            answer = generate_answer(question, chunk)
            if len(answer.strip()) > len(best_answer.strip()):
                best_answer = answer
        print(f"Answer: {best_answer}")
    else:
        print("Sorry, I couldn't find a relevant answer.")

# Take user input and generate answer
question = input("Enter your question: ")
answer_question(question)


# What changes are done by apple recently to improve its market capture?