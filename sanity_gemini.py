# sanity_gemini.py
import os, google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()                                      # loads GOOGLE_API_KEY
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
resp  = model.generate_content("Say 'pong' once.")
print("Model says:", resp.text.strip())