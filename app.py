import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Download required tokenizer
nltk.download('punkt')

# Load model safely on CPU
model = SentenceTransformer('all-MiniLM-L6-v2')
if torch.__version__ >= "2.0.0":
    try:
        model = torch.nn.Module.to_empty(model, device=torch.device("cpu"))
    except Exception:
        pass

# Text analysis
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
        if not transcript_text.strip():
            st.warning("Please enter a transcript.")
            return

        results = dynamic_parameter_tracking(transcript_text)
        st.subheader("Analysis Results")
        st.json(results)

if __name__ == "__main__":
    main()

