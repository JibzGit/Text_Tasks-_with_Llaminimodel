import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer,T5ForConditionalGeneration
from transformers import pipeline
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import base64

# Initialize the model and tokenizer
checkpoint = "Lamini -Flan-T5"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, torch_dtype=torch.float32
)

# Function to preprocess PDF and extract text
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# Summarization pipeline
def summarization_pipeline(filepath):
    pipe_sum = pipeline("summarization", model=base_model, tokenizer=tokenizer, max_length=500, min_length=50)
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    return result[0]["summary_text"]

# Text generation pipeline
def text_generation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = base_model.generate(inputs.input_ids, max_length=200, num_return_sequences=1, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Question-answering pipeline
def question_answering_pipeline(question, context):
    prompt = f"Question: {question} Context: {context} Answer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = base_model.generate(inputs.input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Function to display PDF
@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit app
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization, Text Generation, and Q&A")
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        filepath = "data/" + uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        
        # Display the uploaded PDF
        st.info("Uploaded PDF:")
        displayPDF(filepath)
        
        # Summarization
        if st.button("Summarize"):
            st.info("Summarizing...")
            summary = summarization_pipeline(filepath)
            st.success("Summary:")
            st.write(summary)
        
        # Text Generation
        st.subheader("Text Generation")
        prompt = st.text_input("Enter a prompt for text generation:")
        if st.button("Generate Text"):
            generated_text = text_generation(prompt)
            st.success("Generated Text:")
            st.write(generated_text)
        
        # Question Answering
        st.subheader("Question Answering")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            context = file_preprocessing(filepath)  # Use the full text or summary as context
            answer = question_answering_pipeline(question, context)
            st.success("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()