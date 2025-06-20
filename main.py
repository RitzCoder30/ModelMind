from flask import Flask, render_template, request, session
import pandas as pd
from transformers import pipeline

# Initialize Flask app and secret key for sessions
app = Flask(__name__)
app.secret_key = "replace_with_your_secret_key"

# Initialize LLaMA pipeline once (local or accessible without API key)
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

@app.route('/')
def index():
    # Landing page with file upload and question form
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle initial file upload and first question.
    Saves the summary and conversation history to the session.
    Generates the first answer.
    """
    uploaded_file = request.files.get('file')
    question = request.form.get('question')

    # Basic validation
    if not uploaded_file or uploaded_file.filename == '':
        return "No file uploaded", 400
    if not question:
        return "No question provided", 400

    try:
        # Read Excel file into DataFrame
        df = pd.read_excel(uploaded_file)

        # Create a summary string of the DataFrame
        summary = df.describe(include='all').fillna("").to_string()

        # Store summary and conversation history in session
        session['summary'] = summary
        session['conversation'] = [
            {
                "role": "user",
                "content": f"Summary of financial model:\n{summary}\nQuestion: {question}"
            }
        ]

        # Build prompt for the model
        prompt = (
            "<|begin_of_text|>"
            "<|user|> Here is a summary of a financial model. "
            "Provide a brief, concise, and formal answer (5-6 Sentences Max) to the question:\n"
            f"{summary}\n\nQuestion: {question}\n<|assistant|>"
        )

        # Generate the answer
        result = pipe(prompt, max_new_tokens=300, do_sample=False)
        answer = result[0]['generated_text'].split("<|assistant|>")[-1].strip()

        # Append assistant's answer to conversation history
        session['conversation'].append({"role": "assistant", "content": answer})

        # Render the chat page with the answer and a form for follow-up questions
        return render_template(
            'chat.html',
            answer=answer,
            filename=uploaded_file.filename,
            question=question,
            data_preview=df.head().to_html(index=False)
        )

    except Exception as e:
        return f"Error processing file: {str(e)}", 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle follow-up questions after initial upload.
    Keeps conversation context in the session.
    """
    question = request.form.get('question')
    if not question:
        return "No question provided", 400

    summary = session.get('summary')
    conversation = session.get('conversation', [])

    if not summary or not conversation:
        return "Session expired or no initial data. Please upload the file again.", 400

    # Append user's new question to conversation history
    conversation.append({"role": "user", "content": question})

    # Build prompt string from the conversation history
    prompt = "<|begin_of_text|>\n"
    for msg in conversation:
        if msg["role"] == "user":
            prompt += f"<|user|> {msg['content']}\n"
        else:
            prompt += f"<|assistant|> {msg['content']}\n"
    prompt += "<|assistant|>"

    # Generate new answer
    result = pipe(prompt, max_new_tokens=300, do_sample=False)
    answer = result[0]['generated_text'].split("<|assistant|>")[-1].strip()

    # Update conversation history with assistant's answer
    conversation.append({"role": "assistant", "content": answer})
    session['conversation'] = conversation

    # Render chat page with new answer and follow-up input
    return render_template('chat.html', answer=answer, question=question)


if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=5000)
