import os
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from sentence_transformers import SentenceTransformer

# Ensure NLTK data directory is configured correctly
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_dir)

# Download required NLTK resources if missing
for resource in ['punkt', 'averaged_perceptron_tagger', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'taggers/{resource}' if resource == 'averaged_perceptron_tagger' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# Load the sentence transformer model safely
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error(f"Error loading SentenceTransformer model: {e}")
    st.stop()

# Function to compute average word length
def average_word_length(text):
    words = word_tokenize(text)
    if not words:
        return 0
    return round(sum(len(word) for word in words) / len(words), 2)

# Function to count stopwords
def count_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return sum(1 for word in words if word.lower() in stop_words)

# Function to perform POS tagging
def pos_statistics(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    tags = [tag for _, tag in pos_tags]
    freq_dist = nltk.FreqDist(tags)
    return dict(freq_dist)

# Main analysis function
def dynamic_parameter_tracking(transcript_text):
    return {
        "average_word_length": average_word_length(transcript_text),
        "stopword_count": count_stopwords(transcript_text),
        "pos_distribution": pos_statistics(transcript_text),
        "embedding_dimension": len(model.encode(["test"])[0])
    }

# Streamlit UI
def main():
    st.title("Sales Transcription Analyzer")
    st.markdown("Analyze text from sales transcripts using NLP and embeddings.")
    
    transcript_text = st.text_area("Paste the transcription text below:", height=200)

    if st.button("Run Analysis"):
        if transcript_text.strip():
            results = dynamic_parameter_tracking(transcript_text)
            st.subheader("Analysis Results")
            st.write(results)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()

