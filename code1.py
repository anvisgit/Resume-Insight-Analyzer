import os
import sys
import streamlit as st
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
import spacy
import re

sys.path.append(os.path.abspath("."))
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

def run_ats_checks(resume_text: str):
    warnings = []
    score = 100
    if len(resume_text) < 300:
        warnings.append("Resume seems too short; add more details.")
        score -= 20
    if not re.search(r"\b(email|phone|contact)\b", resume_text, re.I):
        warnings.append("Missing contact info (email or phone).")
        score -= 15
    if not re.search(r"\b(education|degree|university)\b", resume_text, re.I):
        warnings.append("Education section not detected.")
        score -= 15
    if not re.search(r"\b(experience|internship|work)\b", resume_text, re.I):
        warnings.append("Experience section not detected.")
        score -= 20
    return max(score, 0), warnings

st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.title("Resume Analyzer")
st.sidebar.title("Options")
st.sidebar.write("Paste resume and job description below.")

rtxt = st.text_area("Paste Resume Text", height=300)
jtxt = st.text_area("Paste Job Description", height=300)

if rtxt:
    doc = nlp(rtxt)
    resume_skills = [i.text.lower() for i in doc.ents if i.label_ in ["SKILL", "ORG", "PRODUCT"]]
else:
    resume_skills = []

if st.button("Analyze"):
    if not rtxt or not jtxt:
        st.warning("Please paste both resume and job description.")
    else:
        with st.spinner("Analyzing resume..."):
            ats_score, ats_warnings = run_ats_checks(rtxt)
            st.subheader("ATS Compatibility Check")
            st.write(f"ATS Score: {ats_score}/100")
            if ats_warnings:
                st.warning("\n".join(ats_warnings))
            else:
                st.success("Your resume passed common ATS checks.")
            embr = model.encode(rtxt, convert_to_tensor=True)
            embj = model.encode(jtxt, convert_to_tensor=True)
            similarity = util.cos_sim(embr, embj).item()
            st.subheader("Resume–Job Description Similarity")
            st.write(f"Similarity Score: {similarity:.4f}")
            tokenizer = TreebankWordTokenizer()
            tokensr = tokenizer.tokenize(rtxt.lower())
            tokensj = tokenizer.tokenize(jtxt.lower())
            stop_words = set(stopwords.words("english"))
            filtered_tokensr = [w for w in tokensr if w.isalnum() and w not in stop_words]
            filtered_tokensj = [w for w in tokensj if w.isalnum() and w not in stop_words]
            summary_match = re.search(r"(summary|objective)[:\n](.*?)(?:\n\n|\Z)", rtxt, re.I | re.S)
            if summary_match:
                summary_text = summary_match.group(2).strip()
                sentiment = sia.polarity_scores(summary_text)
                st.subheader("Resume Summary Sentiment")
                st.write(summary_text)
                st.write(sentiment)
            jd_words = re.findall(r'\b\w+\b', jtxt.lower())
            jd_skills = [w for w in jd_words if len(w) > 1]
            matched = [s for s in resume_skills if s.lower() in jd_skills]
            missing = [s for s in jd_skills if s.lower() not in resume_skills]
            missing_filtered = [s for s in missing if s.lower() not in stop_words]
            st.subheader("Skill Gap Analysis")
            st.write("Missing skills:", missing_filtered if missing_filtered else "(none)")
            if matched:
                st.write("Matched skills:", matched)
            st.subheader("Suggestions")
            suggestions = []
            if similarity < 0.45:
                suggestions.append("Your resume objective does not match the job description well. Add relevant keywords and experiences from the JD.")
            if missing_filtered:
                suggestions.append(f"Your resume is missing skills required for this role: {', '.join(missing_filtered)}. Add or highlight them.")
            if matched:
                suggestions.append(f"Good! You already have skills that match the JD: {', '.join(matched)}. Make sure they are prominent in your resume.")
            if similarity >= 0.45 and not missing_filtered:
                suggestions.append("Your resume aligns well with the job description. Keep it concise and clear.")
            for s in suggestions:
                st.markdown(f"- {s}")
