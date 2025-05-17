import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU use entirely

import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

# Download tokenizer data
nltk.download('punkt')

# Safely load model WITHOUT .to(device)
model = SentenceTransformer('all-MiniLM-L6-v2')
model._first_module().eval()  # Ensure model is in eval mode without moving device

# Function to calculate average word length
def average_word_length(text):
    words = word_tokenize(text)
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)

# Function to extract metrics
def dynamic_parameter_tracking(text):
    embedding = model.encode([text])[0]
    return {
        "word_count": len(text.split()),
        "average_word_length": average_word_length(text),
        "embedding_vector_mean": float(np.mean(embedding))
    }

# Streamlit UI
def main():
    st.title("Sales Transcript Analyzer")
    transcript_text = st.text_area("Paste the sales transcript below:")

    if st.button("Analyze"):
        if not transcript_text.strip():
            st.warning("Please enter some text.")
            return

        results = dynamic_parameter_tracking(transcript_text)
        st.subheader("Analysis Results")
        st.json(results)

if __name__ == "__main__":
    main()


