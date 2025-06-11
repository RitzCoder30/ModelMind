from flask import Flask, render_template, request
import pandas as pd
import openpyxl

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

        return f"<h3>Received file: {uploaded_file.filename}</h3><h4>Question: {question}</h4>{preview}"

    except Exception as e:
        return f"Error parsing file: {str(e)}", 500