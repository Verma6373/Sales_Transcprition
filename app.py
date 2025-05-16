import os
import re
import string
import time
from collections import Counter

import nltk
import streamlit as st
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textstat import flesch_kincaid_grade

import google.generativeai as genai

# Force SentenceTransformer to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the NLTK data directory
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_dir)

# Ensure required NLTK resources
for resource in ['punkt', 'averaged_perceptron_tagger', 'stopwords']:
    try:
        nltk.data.find(resource if '/' in resource else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# Load the SentenceTransformer model on CPU
model = SentenceTransformer('all-MiniLM-L6-v2')

def read_file(uploaded_file):
    return uploaded_file.getvalue().decode("utf-8")

def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def average_word_length(text):
    words = word_tokenize(text)
    word_lengths = [len(word) for word in words]
    return sum(word_lengths) / len(words)

def punctuation_density(text):
    total_chars = len(text)
    num_punctuation = sum(1 for char in text if char in string.punctuation)
    return num_punctuation / total_chars

def pos_density(text):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    pos_counts = Counter(tag for _, tag in tagged_tokens)
    total_words = len(tokens)
    return {tag: count / total_words for tag, count in pos_counts.items()}

def sentence_complexity(text):
    sentences = sent_tokenize(text)
    return sum(len(sent.split()) for sent in sentences) / len(sentences)

def repetition_ratio(text):
    words = word_tokenize(text)
    unique_words = set(words)
    return (len(words) - len(unique_words)) / len(words)

def dynamic_parameter_tracking(transcript_text):
    return {
        'avg_word_len': average_word_length(transcript_text),
        'punctuation_dens': punctuation_density(transcript_text),
        'pos_dens': pos_density(transcript_text),
        'sent_comp': sentence_complexity(transcript_text),
        'rep_ratio': repetition_ratio(transcript_text),
        'readability_score': flesch_kincaid_grade(transcript_text)
    }

def generate_score_and_justification(transcript_text, avg_word_len, punctuation_dens, pos_dens, sent_comp, rep_ratio, readability_score):
    prompt = f"""
    Analyze the following sales conversation transcript to determine the likelihood of the customer purchasing the course. Provide a score out of 100 for the likelihood of conversion. Also, justify the score with five bullet points...

    Transcript:
    {transcript_text}

    Additional Parameters:
    - Average Word Length: {avg_word_len}
    - Punctuation Density: {punctuation_dens}
    - Part-of-Speech Density: {pos_dens}
    - Sentence Complexity: {sent_comp}
    - Repetition Ratio: {rep_ratio}
    - Readability Score: {readability_score}

    Conversion Score: _______/100
    Justification:
    - Bullet Point 1: ...
    """

    try:
        gemini_model = genai.GenerativeModel('gemini-pro')
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating content: {e}")
        return f"Error generating content: {e}"

def main():
    st.title("Sales Conversation Analysis")

    uploaded_file = st.file_uploader("Upload sales conversation transcript", type=["txt"])
    if uploaded_file:
        content = read_file(uploaded_file)
        cleaned_text = preprocess_text(content)
        parameters = dynamic_parameter_tracking(cleaned_text)
        score_and_justification = generate_score_and_justification(cleaned_text, **parameters)

        st.markdown("**Score and Justification**")
        st.write(score_and_justification)

        # Optional delay to avoid API rate limits
        time.sleep(60)

if __name__ == "__main__":
    main()


