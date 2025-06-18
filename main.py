from flask import Flask, render_template, request
import pandas as pd
from transformers import pipeline

# Initialize LLaMA pipeline
llama = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", device=0)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files.get('file')
    question = request.form.get('question')

    if not uploaded_file or uploaded_file.filename == '':
        return "No file uploaded", 400

    if not question:
        return "No question provided", 400

    try:
        # Read Excel file into DataFrame
        df = pd.read_excel(uploaded_file)

        # Preview the first few rows for the LLaMA prompt
        preview_text = df.head(5).to_string(index=False)

        # Construct prompt for the model
        prompt = (
            f"<|begin_of_text|><|user|> Here is the beginning of a financial model (Excel):\n"
            f"{preview_text}\n\nQuestion: {question}\n<|assistant|>"
        )

        # Generate response
        result = llama(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)

        # Extract and clean the generated answer
        answer = result[0]['generated_text'].split("<|assistant|>")[-1].strip()

        # Return result in HTML
        return (
            f"<h3>File: {uploaded_file.filename}</h3>"
            f"<h4>Question: {question}</h4>"
            f"<p><strong>Answer:</strong> {answer}</p>"
            f"<h4>Data Preview:</h4>{df.head().to_html(index=False)}"
        )

    except Exception as e:
        return f"Error processing file: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
