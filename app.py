import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

# Download necessary tokenizer
nltk.download('punkt')

# Explicitly load model on CPU to avoid NotImplementedError
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Helper functions
def average_word_length(text):
    words = word_tokenize(text)
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)

def dynamic_parameter_tracking(text):
    embedding = model.encode([text])[0]
    return {
        "word_count": len(text.split()),
        "average_word_length": average_word_length(text),
        "embedding_vector_mean": float(np.mean(embedding))
    }

# Streamlit UI
def main():
    st.title("Sales Transcription Analyzer")
    transcript_text = st.text_area("Paste the sales transcript below:")

    if st.button("Analyze"):
        if transcript_text.strip() == "":
            st.warning("Please paste some text to analyze.")
            return

        results = dynamic_parameter_tracking(transcript_text)
        st.subheader("Analysis Results")
        st.json(results)

if __name__ == "__main__":
    main()

