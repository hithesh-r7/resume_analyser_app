import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])

        # Predict probabilities for each category
        probabilities = clf.predict_proba(input_features)[0]

        # Sort probabilities and get top 3 indices
        top_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]

        # Map category ID to category name
        category_mapping = {
            6:"Project Manager",4:"Front End",1:"Backend",3:"Embedded",0:"AI/ML",7:"python developer",5:"HR",2:"data analyst"
        }

        # Display top 3 categories and their probabilities
        st.write("Top 3 suggestions:")
        for i, idx in enumerate(top_indices, start=1):
            category_name = category_mapping.get(idx, "Unknown")
            probability = probabilities[idx]
            st.write(f"{i}. {category_name}:{probability*100:.2f}%")

# python main
if __name__ == "__main__":
    main()
