import streamlit as st
import requests
import tempfile
import os
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"  

# Initialize session state
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

st.set_page_config(page_title="PDF Math & Text Question Answering", layout="wide")
st.title("PDF Question Answering System")

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    process_button = st.button("Process PDF")

if uploaded_file and process_button:
    st.session_state.processing_complete = False

    with st.spinner("Processing PDF with OCR..."):
        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{BACKEND_URL}/process_pdf/", files=files)
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "success":
                    st.session_state.processed_text = result["processed_text"]
                    st.session_state.processing_complete = True
                    st.success("PDF processed successfully! You can now ask questions about it.")
                else:
                    st.error(f"Processing failed: {result.get('message', 'Unknown error')}")
            else:
                st.error(f"Error communicating with backend: {response.text}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if st.session_state.processed_text:
    with st.expander("View Processed Text", expanded=False):
        st.text_area("Extracted and Processed Text", st.session_state.processed_text, height=300)

if st.session_state.processing_complete:
    st.header("Ask Questions")
    user_question = st.text_input("Enter your question about the PDF content:")
    
    if user_question:
        with st.spinner("Processing your question..."):
            try:
                payload = {
                    "question": user_question,
                    "processed_text": st.session_state.processed_text
                }
                response = requests.post(f"{BACKEND_URL}/ask_question/", json=payload)
                
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.subheader("Answer:")
                    st.markdown(answer)
                else:
                    st.error(f"Error getting answer: {response.text}")
                    
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    st.empty()