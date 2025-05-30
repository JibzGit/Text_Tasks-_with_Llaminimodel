import torch
import base64
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Model and tokenizer setup
checkpoint = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)

# 📄 PDF Preprocessing
def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    full_text = "".join([text.page_content for text in texts])
    return full_text

# 📌 Summarization
def summarization_pipeline(file_path):
    full_text = file_preprocessing(file_path)
    summarizer = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=512,
        min_length=50
    )
    result = summarizer(full_text[:3000])  # Truncate for model limits
    return result[0]["summary_text"]

# ✍️ Text Generation
def text_generation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = base_model.generate(inputs.input_ids, max_length=200, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ❓ Question Answering
def question_answering_pipeline(question, context):
    prompt = f"Question: {question} Context: {context} Answer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = base_model.generate(inputs.input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
