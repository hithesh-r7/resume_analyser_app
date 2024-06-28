import streamlit as st
import docx2txt
from pdf2docx import Converter
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def convert_pdf_to_docx(uploaded_file, docx_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    cv = Converter(uploaded_file.name)
    cv.convert(docx_file)


data = [
    ['Xilinx,FPGA,Cadence', 'embedded'],
    ['Java,C++', 'OOPs'],
    ['HTML,CSS,JS', 'Web Dev'],
    ['python,sql,statistics,NLP','dataanalyst']
]
df = pd.DataFrame(data, columns=['text', 'category'])


mandatory_skills = set(["python", "sql", "statistics"])
optional_skills = set(["scikit-learn/TensorFlow", "Hadoop/Spark", "AWS/GCP/Azure", "NLP", "NoSQL/SQL", "Data Scraping",
                       "data preprocessing", "data analysis libraries", "data wrangling and cleaning techniques"])


def main():
    st.title("Resume Ranking for Data Analyst")

    uploaded_files = st.file_uploader("Upload your resumes (PDF)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        ranked_candidates = []
        for idx, uploaded_file in enumerate(uploaded_files):
            st.header(f"Resume {idx + 1}")
            with st.spinner(f"Converting PDF {idx + 1} to DOCX..."):
               
                docx_file = f"uploaded_resume_{idx + 1}.docx"
                convert_pdf_to_docx(uploaded_file, docx_file)

            
            st.spinner(f"Performing similarity search for Resume {idx + 1}...")

            
            resume_text = docx2txt.process(docx_file)

            
            mandatory_skills_present = all(skill in resume_text.lower() for skill in mandatory_skills)
            if not mandatory_skills_present:
                st.write(f"Resume {idx + 1} is rejected due to missing mandatory skills.")
                continue

            
            encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
            resume_embedding = encoder.encode([resume_text])
            category_embeddings = encoder.encode(df['text'])

            
            similarities = cosine_similarity(resume_embedding, category_embeddings)

            
            sorted_indices = np.argsort(similarities[0])[::-1]

            
            optional_skills_present = sum(skill in resume_text.lower() for skill in optional_skills)

            ranked_candidates.append((idx + 1, optional_skills_present))

            st.write(f"Ranked Categories for Resume {idx + 1} by Similarity to Resume Text:")
            for idx_sorted in sorted_indices:
                st.write(f"Category: {df.loc[idx_sorted, 'category']}, Similarity: {similarities[0][idx_sorted]}")

        
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        st.write("Ranking of Candidates based on Number of Optional Skills:")
        for rank, (resume_idx, optional_skills_count) in enumerate(ranked_candidates, start=1):
            st.write(f"Resume {resume_idx}: Rank {rank}, Optional Skills Count: {optional_skills_count}")
if __name__ == "__main__":
    main()
