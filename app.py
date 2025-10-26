# app.py
import os
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader

# Import Hugging Face Transformers
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Hugging Face Summarization Pipeline
# This will download the model the first time it runs.
# 'facebook/bart-large-cnn' is a good general-purpose summarization model.
try:
    summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Hugging Face summarization pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading Hugging Face pipeline: {e}")
    # You might want to handle this error more gracefully in a production app
    summarizer_pipeline = None # Set to None if loading fails

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text

def summarize_text_with_huggingface(text, max_length=150, min_length=40):
    """Summarizes text using a local Hugging Face model."""
    if not text:
        return "No text provided for summarization."
    if summarizer_pipeline is None:
        return "Summarization model not loaded. Please check server logs."

    # Hugging Face models have a context window limit (e.g., BART is 1024 tokens).
    # For very long documents, you might need more sophisticated chunking,
    # but for typical use cases, truncating is often sufficient for summarization.
    # The pipeline handles tokenization internally.
    max_model_input_length = summarizer_pipeline.model.config.max_position_embeddings
    
    # Simple truncation for demonstration.
    # A more robust solution for very long docs would involve splitting the text,
    # summarizing chunks, and then summarizing the summaries.
    if len(text.split()) > (max_model_input_length * 0.75): # Estimate based on words vs tokens
        text = " ".join(text.split()[:int(max_model_input_length * 0.75)])


    try:
        # The pipeline returns a list of dictionaries. We want the 'summary_text' from the first element.
        summary = summarizer_pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False # For more deterministic output
        )
        return summary[0]['summary_text'].strip()

    except Exception as e:
        print(f"Hugging Face summarization error: {e}")
        return f"Summarization error: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        extracted_text = extract_text_from_pdf(file_path)
        os.remove(file_path)  # Clean up after processing

        if extracted_text:
            summary = summarize_text_with_huggingface(extracted_text)
            return jsonify({"summary": summary})
        else:
            return jsonify({"error": "Could not extract text from PDF or PDF was empty."}), 500

@app.route('/summarize_text', methods=['POST'])
def summarize_text_from_input():
    data = request.get_json()
    text_input = data.get('text')

    if not text_input:
        return jsonify({"error": "No text provided"}), 400

    summary = summarize_text_with_huggingface(text_input)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)