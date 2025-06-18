from flask import Flask, render_template, request
import pandas as pd
import openpyxl
import transformers
from transformers import pipeline

llama = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", device=0)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files.get('file')
    question = request.form.get('question')

    if not uploaded_file or uploaded_file.filename == '':
        return "No file uploaded", 400

    if not question:
        return "No question provided", 400
    # Read the Excel file
    try:
        # Read the uploaded Excel file into a pandas DataFrame
        df = pd.read_excel(uploaded_file)

        # Display the first few rows for demo purposes
        preview = df.head().to_html()

        prompt = f"""<|begin_of_text|><|user|>Here is the beginning of the financial model (Excel): {preview_text}
        Question: {question}
        <|assistant|>"""
            # Generate a response using LLaMA
        result = llama(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
        answer = result[0]['generated_text'].split("<|assistant|>")[-1].strip()

        return f"<h3>File: {uploaded_file.filename}</h3><h4>Question: {question}</h4><p><strong>Answer:</strong> {answer}</p><h4>Data Preview:</h4>{df.head().to_html()}"

    except Exception as e:
        return f"Error parsing file: {str(e)}", 500