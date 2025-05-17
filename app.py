import os
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from sentence_transformers import SentenceTransformer

# Configure NLTK data path
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_dir)

# Function to safely download NLTK data
def download_nltk_data():
    required_resources = {
        "punkt": "tokenizers/punkt",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "stopwords": "corpora/stopwords"
    }

    for name, path in required_resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, download_dir=nltk_data_dir)

download_nltk_data()

# Load the SentenceTransformer model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# NLP analysis functions
def average_word_length(text):
    words = word_tokenize(text)
    return round(sum(len(word) for word in words) / len(words), 2) if words else 0

def count_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    return sum(1 for word in words if word.lower() in stop_words)

def pos_statistics(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    tag_freq = nltk.FreqDist(tag for _, tag in pos_tags)
    return dict(tag_freq)

def dynamic_parameter_tracking(transcript_text):
    return {
        "average_word_length": average_word_length(transcript_text),
        "stopword_count": count_stopwords(transcript_text),
        "pos_distribution": pos_statistics(transcript_text),
        "embedding_dimension": len(model.encode(["sample text"])[0])
    }

# Streamlit interface
def main():
    st.title("Sales Transcription Analysis")
    st.markdown("Analyze your sales transcription using NLP and embeddings.")

    transcript_text = st.text_area("Paste the transcription text below:")

    if st.button("Analyze"):
        if transcript_text.strip():
            try:
                results = dynamic_parameter_tracking(transcript_text)
                st.subheader("Analysis Results")
                st.json(results)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
        else:
            st.warning("Please enter transcription text.")

if __name__ == "__main__":
    main()

