import streamlit as st
from utils import extract_text_from_pdf, preprocess_text
from models import load_qa_model, get_answer

st.title("Enhanced PDF Question Answering App")

st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    
    st.sidebar.success("PDF text extracted successfully!")

    with st.spinner("Preprocessing text..."):
        processed_text = preprocess_text(pdf_text)
    
    st.sidebar.success("Text preprocessed successfully!")

    st.header("Ask a Question")
    question = st.text_input("Enter your question:")
    
    if question:
        with st.spinner("Loading QA model and getting answer..."):
            qa_model, summarizer = load_qa_model()
            answer = get_answer(qa_model, summarizer, processed_text, question)
        
        st.write("**Answer:**", answer)
else:
    st.sidebar.info("Please upload a PDF file to proceed.")
