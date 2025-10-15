import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tempfile
import os
import spacy
from nltk.tokenize import TreebankWordTokenizer
import re 
from PyPDF2 import PdfReader
import docx

nltk.download('punkt', quiet=True)
nltk.download("stopwords")
nltk.download("vader_lexicon")#sentiment word dictionary 

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm") #english, core, trained on web, smallmodel

st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.sidebar.title("Resume Analyzer")
st.sidebar.write("introbs")

uploaded_resume = st.file_uploader("Upload Resume File", type=["pdf", "docx", "txt"])
jtxt = st.text_area("Paste Job Description", height=350)

def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
    return text.strip()

rtxt = ""
if uploaded_resume:
    rtxt = extract_text(uploaded_resume)
    st.text_area("Resume Text", rtxt, height=350)

if rtxt:
    doc = nlp(rtxt)
    resume_skills = [i.text.lower() for i in doc.ents if i.label_ in ["SKILL", "ORG", "PRODUCT"]]

if st.button("Analyze"):
    if not rtxt or not jtxt:
        st.warning("Please paste both resume and job description.")
    else:
         with st.spinner("Computing embeddings"):
            embr = model.encode(rtxt, convert_to_tensor=True)
            embj = model.encode(jtxt, convert_to_tensor=True)
            similarity = util.cos_sim(embr, embj).item()

            st.success(f"Resume–Job Description Similarity Score: **{similarity:.4f}**")
            tokenizer = TreebankWordTokenizer()
            tokensr = tokenizer.tokenize(rtxt.lower())
            tokensj=tokenizer.tokenize(jtxt.lower())
            stop_words = set(stopwords.words("english"))
            filtered_tokensr = [w for w in tokensr if w.isalnum() and w not in stop_words]
            filtered_tokensj = [w for w in tokensj if w.isalnum() and w not in stop_words]

            sentiment = sia.polarity_scores(rtxt)
            st.subheader("Overall Resume Sentiment")
            st.write(sentiment)

            jd_words = re.findall(r'\b\w+\b', jtxt.lower()) #splits and lowers 
            jd_skills = [w for w in jd_words if len(w) > 1]  
            matched = [s for s in resume_skills if s.lower() in jd_skills]
            missing = [s for s in jd_skills if s.lower() not in resume_skills]
            missingfil=[s for s in missing if s.lower() not in stop_words]
            st.write("Missing skills:", missingfil if missingfil else "(none)")

            st.subheader("Suggestions")
            suggestions = []

            if similarity < 0.45:
                suggestions.append("Your resume objective doesn’t match the job description well. Consider adding relevant keywords and experiences from the JD.")

            if missing:
                suggestions.append(f"Your resume is missing skills required for this role: {', '.join(missing)}. Consider adding or highlighting them.")

            if matched:
                suggestions.append(f"Good! You already have skills that match the JD: {', '.join(matched)}. Make sure they are prominent in your resume.")

            if similarity >= 0.45 and not missing:
                suggestions.append("Your resume aligns well with the job description. Keep it concise and clear!")

            for s in suggestions:
                st.markdown(f"- {s}")
