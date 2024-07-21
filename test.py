import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def test_google_api():
    model = genai.GenerativeModel('gemini-pro')
    prompt = "Test the Google Generative AI API with this prompt."
    
    try:
        response = model.generate_content(prompt)
        print("API call successful! Response:")
        print(response.text.strip())
    except Exception as e:
        print(f"API call failed: {e}")

if __name__ == "__main__":
    test_google_api()
