# Resume Insight Analyzer
A Streamlit-based AI app that analyzes your resume against a given job description using **Sentence Transformers**, **SpaCy**, and **Sentiment Analysis**.  
It computes semantic similarity, performs sentiment analysis, and identifies missing or matching skills to help you tailor your resume for specific roles.

---

## Features

- Upload PDF, DOCX, or TXT files; the app will extract text.  
- Uses the `all-MiniLM-L6-v2` model from Sentence Transformers to compare resume and job description.  
- Sentiment Analysis – Evaluates overall tone of your resume.  
- Skill Matching – Highlights overlapping and missing keywords between your resume and job description.  
- Provides targeted feedback to improve alignment with job requirements.

---

## Python Stack

- Streamlit
- Sentence Transformers
- SpaCy
- NLTK
- PyPDF2
- python-docx
