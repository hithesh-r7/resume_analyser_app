import streamlit as st
import docx2txt
from pdf2docx import Converter
import pandas as pd
import PyPDF2
import pyperclip
import csv
from groq import Groq
import tempfile
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def generate_summary(description):
    groq_api = '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'
    client = Groq(api_key=groq_api)
    summary = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {
                "role": "system",
                "content": "You are supposed to summarize the resume of an applicant who has applied for the role of data analyst in 100 words.\n Try to provide a concise and crisp summary mainly on the technologies and the key projects of the applicant.\n If the person's job skills match with the role of data analyst then always start with the <name of applicant> is suitable for the role of Data Analyst mentioned in the description.\n If the skills of the applicant do not match the the role of data analyst then elaborate on the skills that the applicant needs to learn and work upon on if he is recruited as a data analyst in our organization "
            },
            {
                "role": "user",
                "content": description
            },
        ],
        temperature=0.4,
        max_tokens=200,
        top_p=1,
        stream=False,
        stop=None,
    )
    summary = summary.choices[0].message.content
    return summary

def convert_pdf_to_docx(uploaded_file, docx_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    cv = Converter(uploaded_file.name)
    cv.convert(docx_file)

def create_csv(pdf_files):
    with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Id', 'Description', 'Summary']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for idx, file in enumerate(pdf_files, start=1):
            # Save uploaded PDF to a temporary file
            temp_pdf = tempfile.NamedTemporaryFile(delete=False)
            temp_pdf.write(file.read())
            temp_pdf.close()
            
            # Convert PDF to DOCX
            docx_file = f"uploaded_resume_{idx}.docx"
            convert_pdf_to_docx(file, docx_file)
            
            # Extract text from the temporary PDF file
            extracted_text = extract_text_from_pdf(temp_pdf.name)
            
            # Generate summary
            summary = generate_summary(extracted_text)
            
            # Write to CSV
            writer.writerow({'Id': idx, 'Description': extracted_text, 'Summary': summary})
            
            # Remove temporary files
            os.unlink(temp_pdf.name)
            os.unlink(docx_file)

def main():
    st.title("Resume Ranking and Summary Generator")

    uploaded_files = st.file_uploader("Upload PDF files containing resumes", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Rank resumes based on content similarity
        ranked_candidates = []
        for idx, uploaded_file in enumerate(uploaded_files):
            st.header(f"Resume {idx + 1}")
            
            st.spinner(f"Converting PDF {idx + 1} to DOCX...")
            docx_file = f"uploaded_resume_{idx + 1}.docx"
            convert_pdf_to_docx(uploaded_file, docx_file)
            
            st.spinner(f"Performing similarity search for Resume {idx + 1}...")
            resume_text = docx2txt.process(docx_file)
            
            encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
            resume_embedding = encoder.encode([resume_text])
            category_embeddings = encoder.encode(df['text'])
            
            similarities = cosine_similarity(resume_embedding, category_embeddings)
            sorted_indices = np.argsort(similarities[0])[::-1]
            
            ranked_candidates.append((idx + 1, similarities[0][sorted_indices[0]]))
            st.write(f"Ranked Categories for Resume {idx + 1} by Similarity to Resume Text:")
            for idx_sorted in sorted_indices:
                st.write(f"Category: {df.loc[idx_sorted, 'category']}, Similarity: {similarities[0][idx_sorted]}")
        
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        st.write("Ranking of Candidates based on Similarity:")
        for rank, (resume_idx, similarity_score) in enumerate(ranked_candidates, start=1):
            st.write(f"Resume {resume_idx}: Rank {rank}, Similarity Score: {similarity_score}")

        # Generate summaries for top-ranked resumes
        st.write("Generating Summaries for Top-ranked Resumes:")
        for rank, (resume_idx, similarity_score) in enumerate(ranked_candidates[:3], start=1):
            st.write(f"Summary for Resume {resume_idx}:")
            with open(f"uploaded_resume_{resume_idx}.docx", "rb") as docx_file:
                extracted_text = docx2txt.process(docx_file)
                summary = generate_summary(extracted_text)
                st.write(summary)

if __name__ == "__main__":
    # Load category data
    data = [
        ['Xilinx,FPGA,Cadence', 'embedded'],
        ['Java,C++', 'OOPs'],
        ['HTML,CSS,JS', 'Web Dev'],
        ['python,sql,statistics,NLP','dataanalyst']
    ]
    df = pd.DataFrame(data, columns=['text', 'category'])
    
    main()
