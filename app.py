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
from sentence_transformers import SentenceTransformer
from textstat import flesch_kincaid_grade

import google.generativeai as genai
import torch

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the NLTK data directory and ensure it is used
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_dir)

# Ensure NLTK data is downloaded
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

# Debugging to ensure required NLTK data is available
print("NLTK data path:", nltk.data.path)

try:
    nltk.data.find('tokenizers/punkt')
    print("Punkt tokenizer found.")
except LookupError:
    print("Punkt tokenizer not found. Attempting to download...")
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    print("Averaged perceptron tagger found.")
except LookupError:
    print("Averaged perceptron tagger not found. Attempting to download...")
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
    print("Stopwords corpus found.")
except LookupError:
    print("Stopwords corpus not found. Attempting to download...")
    nltk.download('stopwords', download_dir=nltk_data_dir)

# Initialize the Sentence Transformer model with proper device handling
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def read_file(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8")
    return content

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
    num_punctuation = sum([1 for char in text if char in string.punctuation])
    return num_punctuation / total_chars

def pos_density(text):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    pos_counts = Counter(tag for word, tag in tagged_tokens)
    total_words = len(tokens)
    pos_density = {tag: count / total_words for tag, count in pos_counts.items()}
    return pos_density

def sentence_complexity(text):
    sentences = sent_tokenize(text)
    complexity = sum([len(sent.split()) for sent in sentences]) / len(sentences)
    return complexity

def repetition_ratio(text):
    words = word_tokenize(text)
    unique_words = set(words)
    repetition_ratio = (len(words) - len(unique_words)) / len(words)
    return repetition_ratio

def dynamic_parameter_tracking(transcript_text):
    avg_word_len = average_word_length(transcript_text)
    punctuation_dens = punctuation_density(transcript_text)
    pos_dens = pos_density(transcript_text)
    sent_comp = sentence_complexity(transcript_text)
    rep_ratio = repetition_ratio(transcript_text)
    readability_score = flesch_kincaid_grade(transcript_text)

    updated_parameters = {
        'avg_word_len': avg_word_len,
        'punctuation_dens': punctuation_dens,
        'pos_dens': pos_dens,
        'sent_comp': sent_comp,
        'rep_ratio': rep_ratio,
        'readability_score': readability_score
    }

    return updated_parameters

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

    (The rest of the prompt continues here...)
    """

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating content: {e}")
        return f"Error generating content: {e}"

def main():
    st.title("Sales Conversation Analysis")

    uploaded_file = st.file_uploader("Upload sales conversation transcript", type=["txt"])
    if uploaded_file is not None:
        content = read_file(uploaded_file)
        cleaned_text = preprocess_text(content)
        updated_parameters = dynamic_parameter_tracking(cleaned_text)

        # Generate score and justification
        score_and_justification = generate_score_and_justification(cleaned_text, **updated_parameters)

        # Display results
        st.markdown("**Score and Justification**")
        st.write(score_and_justification)

        # Add a delay between requests to avoid hitting the rate limit
        time.sleep(60)

if __name__ == "__main__":
    main()


