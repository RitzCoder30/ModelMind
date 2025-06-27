from flask import Flask, render_template, request, session
import pandas as pd
from transformers import pipeline
import io, os
from openpyxl import load_workbook
from difflib import get_close_matches
import logging
from uuid import uuid4
import spacy

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key")
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload

# Initialize the text generation pipeline. Using a smaller, faster model for this task.
pipe = pipeline("text2text-generation", model="google/flan-t5-small")

# Define common Excel error strings that should be ignored during data extraction
errors = {"#DIV/0!", "#VALUE!", "#REF!", "#NAME?", "#NUM!", "#N/A", "#NULL!"}

# In-memory storage for uploaded file bytes, keyed by session ID.
# This allows the app to retain the DataFrame in memory for subsequent chat queries
# without re-reading the file from disk (or re-uploading).
uploaded_files = {}

# Configure logging for better visibility into application flow and potential issues
logging.basicConfig(level=logging.INFO)

# Load the spaCy English NLP model for natural language understanding.
# This model is used to parse user questions and identify key entities like metrics, departments, and years.
nlp = spacy.load("en_core_web_sm")


# --- Utilities ---

def normalize_col(s):
    return (
        str(s)
        .strip()
        .lower()
        .replace("&", "and")
        .replace(".", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def dataframe_to_text_table(df):
    # Create a copy, fill any NaN values with empty strings, and convert all data to string type
    df = df.copy().fillna("").astype(str)
    headers = list(df.columns)
    # Create the header row and a separator row for markdown table formatting
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    # Add each row's data to the table
    for _, row in df.iterrows():
        lines.append(" | ".join(row.values))
    return "\n".join(lines)


def fuzzy_match(label, aliases, cutoff=0.8):
    return any(get_close_matches(label.lower(), [alias.lower() for alias in aliases], n=1, cutoff=cutoff))


def extract_financial_summary(df):
    df = df.copy()
    # Normalize all column names for consistent access
    df.columns = [normalize_col(col) for col in df.columns]

    label_col = df.columns[
        0]  # Assumes the first column contains the financial metric labels (e.g., "Category_Revenue")

    # Attempt to find a column explicitly named 'total' or containing 'total'
    total_col = next((col for col in df.columns if 'total' in col), None)

    # Identify other columns that might represent departmental or year-based breakdowns
    # These are columns that are not the label column and not the identified total column
    dept_cols = [col for col in df.columns if col not in {label_col, total_col}]

    # Canonical mapping of financial metrics to their common aliases
    row_label_map = {
        "Revenue": ["Revenue", "Total Revenue", "Revenues", "Sales", "Category_Revenue"],
        "Cost of Goods Sold": ["COGS", "Cost of Goods Sold"],
        "Gross Profit": ["Gross Profit", "Gross Margin"],
        "Operating Expenses": ["Operating Expenses", "OPEX", "Operating Costs"],
        "Operating Profit": ["Operating Profit", "Operating Income", "EBIT"],
        "Net Income": ["Net Income", "Net Profit"]
    }

    extracted = {}  # Dictionary to store extracted financial figures

    # Iterate through each row of the DataFrame to find and extract metric values
    for _, row in df.iterrows():
        label = str(row[label_col]).strip()
        # Attempt to match the row label to one of the canonical financial metrics
        for canonical, aliases in row_label_map.items():
            if fuzzy_match(label, aliases):
                total_val = None
                if total_col and total_col in row and pd.notnull(row[total_col]) and row[total_col] not in errors:
                    try:
                        total_val = float(row[total_col])
                    except ValueError:
                        logging.info(f"Skipping non-numeric total value for {label}: {row[total_col]}")

                if total_val is not None:
                    depts = []
                    # Collect breakdown values from department/year columns
                    for dept_col in dept_cols:
                        val = row.get(dept_col)
                        if pd.notnull(val) and val not in errors:
                            try:
                                val = float(val)
                                # Format breakdown for summary, e.g., "Sales_2023: $100.00"
                                depts.append(f"{dept_col.replace('_', ' ').title()}: ${val:,.2f}")
                            except ValueError:
                                logging.info(f"Skipping non-numeric value for {dept_col} under {label}: {val}")
                                continue
                    extracted[canonical] = (total_val, depts)
                break  # Move to the next row once a metric match is found

    if not extracted:
        return "Could not extract key financial figures."

    # Format the extracted figures into a human-readable summary string
    lines = []
    for k, (total, dept_vals) in extracted.items():
        if dept_vals:
            lines.append(f"- {k}: ${total:,.2f} ({'; '.join(dept_vals)})")
        else:
            lines.append(f"- {k}: ${total:,.2f}")
    return "Key Financial Figures:\n" + "\n".join(lines)


def extract_detailed_summaries(df):

    df_processed = df.copy()
    # Normalize all column names consistently
    df_processed.columns = [normalize_col(col) for col in df.columns]
    logging.info(f"Columns after normalization in extract_detailed_summaries: {df.columns.tolist()}")

    label_col = df_processed.columns[0]

    # Canonical mapping for financial metrics
    metric_aliases = {
        "revenue": ["revenue", "total_revenue", "revenues", "sales"],
        "cost_of_goods_sold": ["cogs", "cost of goods sold", "cost_of_goods_sold"],
        "gross_profit": ["gross profit", "gross_margin"],
        "operating_expenses": ["operating_expenses", "opex", "operating_costs"],
        "operating_profit": ["operating_profit", "operating_income", "ebit"],
        "net_income": ["net_income", "net_profit"]
    }

    detailed_summary = {}  # The main dictionary to store the structured data

    # --- Special Handling for REVENUE based on observed structure ---
    # Revenue data is in column headers.
    detailed_summary['revenue'] = {}

    for col_name in df_processed.columns:
        if 'total' in col_name:
            try:
                total_val_str = col_name.split('total_')[-1]
                total_val = float(total_val_str)
                detailed_summary['revenue']['total'] = total_val
                logging.info(f"Extracted total revenue from column name '{col_name}': ${total_val:,.2f}")
            except (ValueError, IndexError):
                logging.warning(f"Could not extract numeric total from column name: {col_name}")

        elif 'dept' in col_name and '_' in col_name:
            try:
                parts = col_name.rsplit('_', 1)
                if len(parts) == 2:
                    dept_name = parts[0]
                    dept_val_str = parts[1]
                    dept_val = float(dept_val_str)

                    # Store directly under department name for simplicity
                    detailed_summary['revenue'][dept_name] = dept_val
                    logging.info(f"Extracted revenue for '{dept_name}' from column name '{col_name}': ${dept_val:,.2f}")
            except (ValueError, IndexError):
                logging.warning(f"Could not extract numeric department revenue from column name: {col_name}")

    # --- General Handling for other metrics (Cost of Goods Sold, Gross Profit, etc.) from rows ---
    for _, row in df_processed.iterrows():
        row_label = str(row[label_col]).strip()
        logging.info(f"Processing row label: '{row_label}' from column '{label_col}' for general metrics.")
        matched_metric = None

        for canonical_metric, aliases in metric_aliases.items():
            if canonical_metric != "revenue" and fuzzy_match(row_label, aliases):
                matched_metric = canonical_metric
                logging.info(f"Matched '{row_label}' to canonical metric: '{canonical_metric}'")
                break

        if matched_metric:
            if matched_metric not in detailed_summary:
                detailed_summary[matched_metric] = {}

            for col_name in df_processed.columns:
                if col_name == label_col:
                    continue

                value = row.get(col_name)
                if pd.notnull(value) and value not in errors:
                    try:
                        float_value = float(value)
                    except ValueError:
                        logging.info(
                            f"Skipping non-numeric value '{value}' in column '{col_name}' for metric '{matched_metric}'.")
                        continue

                    if 'total' in col_name:
                        detailed_summary[matched_metric]['total'] = float_value
                        continue

                    parts = col_name.split('_')
                    if len(parts) >= 2:
                        parent_key = parts[0]
                        child_key = "_".join(parts[1:])

                        if parent_key not in detailed_summary[matched_metric]:
                            detailed_summary[matched_metric][parent_key] = {}
                        detailed_summary[matched_metric][parent_key][child_key] = float_value
                    else:
                        detailed_summary[matched_metric][col_name] = float_value

    logging.info(f"Detailed summary built: {detailed_summary}")
    return detailed_summary


def get_error_cells(file_bytes):
    """
    Loads an Excel workbook from bytes and identifies cells that contain
    known Excel error strings (e.g., "#DIV/0!").
    Returns a list of cell coordinates (e.g., "A1", "B5") where errors are found.
    """
    workbook = load_workbook(filename=io.BytesIO(file_bytes), data_only=False)
    sheet = workbook.active
    return [
        cell.coordinate
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str) and cell.value.strip() in errors
    ]


def query_detailed_summary(question, detailed_summary):
    """
    Parses a natural language question using NLP to extract a requested financial metric,
    potential parent category (e.g., department name), and child category (e.g., year).
    It then queries the `detailed_summary` structure to retrieve the most specific answer.
    """
    doc = nlp(question.lower())

    # Keywords for identifying financial metrics in the question
    metric_keywords = {
        "revenue": ["revenue", "sales", "income", "revenues"],
        "cost_of_goods_sold": ["cogs", "cost of goods sold", "cost_of_goods_sold"],
        "gross_profit": ["gross profit", "margin", "gross_margin"],
        "operating_expenses": ["operating cost", "opex", "operating expenses", "operating_costs"],
        "operating_profit": ["operating profit", "operating income", "ebit"],
        "net_income": ["net income", "net profit"]
    }

    requested_metric = None
    # Identify the main financial metric requested in the question
    for token in doc:
        for canonical, aliases in metric_keywords.items():
            if fuzzy_match(token.text, aliases, cutoff=0.85):  # Slightly adjusted cutoff for better metric detection
                requested_metric = canonical
                break
        if requested_metric:
            break

    if not requested_metric:
        for chunk in doc.noun_chunks:  # Check noun chunks for multi-word metrics
            for canonical, aliases in metric_keywords.items():
                if fuzzy_match(chunk.text, aliases, cutoff=0.85):  # Slightly adjusted cutoff
                    requested_metric = canonical
                    break
            if requested_metric:
                break

    logging.info(f"NLP detected requested metric: {requested_metric}")

    # If no metric is identified or the metric isn't in the summary, return None
    if not requested_metric or requested_metric not in detailed_summary:
        logging.info(f"No valid metric found or metric '{requested_metric}' not in summary for question: {question}")
        return None

    # Collect all possible parent and child keys from the detailed_summary for NLP matching.
    # This makes the matching dynamic based on the actual data columns.
    possible_parent_keys = set()
    possible_child_keys = set()  # Child keys are only relevant if parents have nested dicts
    for metric_data_val in detailed_summary.values():
        for key, value in metric_data_val.items():
            if isinstance(value, dict):  # If the value is a dictionary, its key is a parent category
                possible_parent_keys.add(key)
                for child_key in value.keys():  # Keys within this dictionary are child categories
                    possible_child_keys.add(child_key)
            elif key != 'total' and not key.startswith('category_'):  # Direct key, not 'total' or 'category_xx'
                # This handles cases like 'sales_dept' directly holding a value,
                # so 'sales_dept' acts as a 'parent' for lookup, but doesn't have nested children.
                # Adding it to possible_parent_keys.
                possible_parent_keys.add(key)

    logging.info(f"Possible parent keys from data: {possible_parent_keys}")
    logging.info(f"Possible child keys from data: {possible_child_keys}")

    parent_match_from_nlp = None  # The parent key detected by NLP from the question
    child_match_from_nlp = None  # The child key detected by NLP from the question

    # Try to match parent and child from the question
    for chunk in doc.noun_chunks:
        chunk_text = normalize_col(chunk.text)

        match_p_list = get_close_matches(chunk_text, list(possible_parent_keys), n=1, cutoff=0.75)
        if match_p_list:
            matched_key = match_p_list[0]
            if not parent_match_from_nlp or len(matched_key) > len(parent_match_from_nlp):
                parent_match_from_nlp = matched_key
                logging.info(
                    f"NLP matched parent: '{parent_match_from_nlp}' from chunk: '{chunk.text}' (normalized: '{chunk_text}')")

        match_c_list = get_close_matches(chunk_text, list(possible_child_keys), n=1, cutoff=0.75)
        if match_c_list:
            matched_key = match_c_list[0]
            if not child_match_from_nlp or len(matched_key) > len(child_match_from_nlp):
                child_match_from_nlp = matched_key
                logging.info(
                    f"NLP matched child: '{child_match_from_nlp}' from chunk: '{chunk.text}' (normalized: '{chunk_text}')")

    for token in doc:
        token_text = normalize_col(token.text)

        match_p_list = get_close_matches(token_text, list(possible_parent_keys), n=1, cutoff=0.8)
        if match_p_list:
            matched_key = match_p_list[0]
            if not parent_match_from_nlp or len(matched_key) > len(parent_match_from_nlp):
                parent_match_from_nlp = matched_key
                logging.info(
                    f"NLP matched parent (token): '{parent_match_from_nlp}' from token: '{token.text}' (normalized: '{token_text}')")

        match_c_list = get_close_matches(token_text, list(possible_child_keys), n=1, cutoff=0.8)
        if match_c_list:
            matched_key = match_c_list[0]
            if not child_match_from_nlp or len(matched_key) > len(child_match_from_nlp):
                child_match_from_nlp = matched_key
                logging.info(
                    f"NLP matched child (token): '{child_match_from_nlp}' from token: '{token.text}' (normalized: '{token_text}')")

    logging.info(f"Final NLP matches: Parent='{parent_match_from_nlp}', Child='{child_match_from_nlp}'")

    metric_data = detailed_summary[requested_metric]
    logging.info(f"Data for requested metric '{requested_metric}': {metric_data}")

    if parent_match_from_nlp and child_match_from_nlp:
        logging.info(f"Attempting to retrieve with parent='{parent_match_from_nlp}' and child='{child_match_from_nlp}'")
        if parent_match_from_nlp in metric_data and isinstance(metric_data[parent_match_from_nlp], dict) and \
                child_match_from_nlp in metric_data[parent_match_from_nlp]:
            val = metric_data[parent_match_from_nlp][child_match_from_nlp]
            logging.info(f"Found value: ${val:,.2f}")
            return f"${val:,.2f}"
        logging.info("Parent-child path not found directly for nested dictionary.")

    # Case 2: Only parent is matched (e.g., "revenue for R&D department")
    if parent_match_from_nlp:
        logging.info(f"Attempting to retrieve with only parent='{parent_match_from_nlp}'")
        # Find the *actual* key in metric_data that fuzzily matches parent_match_from_nlp
        actual_parent_key_in_data = None
        # Lowered cutoff to 0.6 for more lenient fuzzy matching of NLP identified parent to data keys
        match_list_for_data_key = get_close_matches(parent_match_from_nlp, list(metric_data.keys()), n=1, cutoff=0.6)
        if match_list_for_data_key:
            actual_parent_key_in_data = match_list_for_data_key[0]
            logging.info(
                f"Fuzzy matched NLP parent '{parent_match_from_nlp}' to actual data key '{actual_parent_key_in_data}'")

        if actual_parent_key_in_data and actual_parent_key_in_data in metric_data:
            data = metric_data[actual_parent_key_in_data]
            if isinstance(data, (
            int, float)):  # If parent key directly holds the value (like 'revenue': {'randd_dept': 12000.0})
                logging.info(f"Found direct parent value: ${data:,.2f}")
                return f"${data:,.2f}"
            elif isinstance(data,
                            dict):  # If parent is a dictionary of child values, sum them up (e.g., 'sales': {'dept_80000': 32000.0})
                total_for_parent = sum(v for v in data.values() if isinstance(v, (int, float)))
                if total_for_parent > 0:
                    logging.info(f"Summed values for parent '{actual_parent_key_in_data}': ${total_for_parent:,.2f}")
                    return f"${total_for_parent:,.2f}"
        logging.info("Parent match yielded no direct or summable value for metric.")

    # Case 3: Only child is matched (e.g., "total for 2023" assuming '2023' is a child key across parents)
    if child_match_from_nlp:
        logging.info(f"Attempting to retrieve with only child='{child_match_from_nlp}'")
        # Check if the child_match is a direct key in the metric data
        if child_match_from_nlp in metric_data and isinstance(metric_data[child_match_from_nlp], (int, float)):
            val = metric_data[child_match_from_nlp]
            logging.info(f"Found direct child value: ${val:,.2f}")
            return f"${val:,.2f}"

        # Otherwise, iterate through all parent categories to find and sum values for this specific child
        sum_for_child = 0
        found_child_in_parent = False
        for key, value_obj in metric_data.items():
            if isinstance(value_obj, dict) and child_match_from_nlp in value_obj and isinstance(
                    value_obj[child_match_from_nlp], (int, float)):
                sum_for_child += value_obj[child_match_from_nlp]
                found_child_in_parent = True
        if found_child_in_parent:
            logging.info(f"Summed values for child '{child_match_from_nlp}' across parents: ${sum_for_child:,.2f}")
            return f"${sum_for_child:,.2f}"
        logging.info("Child match yielded no direct or summable value for metric.")

    # Case 4: Question asks for a 'total' or 'overall' value without specific breakdowns
    if any(word in doc.text for word in
           ["total", "overall", "all", "sum", "company"]):  # Added 'company' for total profit
        logging.info("Question contains 'total' or similar keyword. Attempting to find overall total.")
        if 'total' in metric_data and isinstance(metric_data['total'], (int, float)):
            logging.info(f"Found explicit 'total' key: ${metric_data['total']:,.2f}")
            return f"${metric_data['total']:,.2f}"

        # If no explicit 'total' key, calculate the sum of all direct numeric values and sums from first-level dictionaries
        total_sum = 0
        for key, value_obj in metric_data.items():
            if isinstance(value_obj, (
            int, float)) and key != 'total':  # Sum direct numeric values (excluding 'total' itself if it exists)
                total_sum += value_obj
            elif isinstance(value_obj, dict):  # Sum values within nested dictionaries
                total_sum += sum(v for v in value_obj.values() if isinstance(v, (int, float)))

        if total_sum > 0:  # Ensure sum is meaningful
            logging.info(f"Calculated overall sum for '{requested_metric}': ${total_sum:,.2f}")
            return f"${total_sum:,.2f}"
        logging.info("No overall total found or calculable.")

    logging.info(
        f"No specific answer found for question: '{question}' after querying detailed summary. Falling back to LLM.")
    return None  # If no direct answer can be found, return None to trigger LLM fallback


# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files.get('file')
    question = request.form.get('question')

    if not uploaded_file or not question:
        return "Missing file or question", 400

    try:
        # Assign a unique session ID to keep track of uploaded files and conversation history
        session_id = session.setdefault('session_id', str(uuid4()))
        file_bytes = uploaded_file.read()
        uploaded_files[session_id] = file_bytes  # Store file bytes in-memory

        # Read Excel file with multi-level headers (header=[0, 1]) and flatten column names.
        # This handles the "nested tables" by creating columns like 'Sales_2023'.
        df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl', header=[0, 1])
        # Join multi-index columns with an underscore. Handle cases where parts might be None/NaN.
        df.columns = ['_'.join(filter(None, [str(i) for i in col])) for col in df.columns]
        logging.info(f"Flattened DataFrame columns after upload: {df.columns.tolist()}")
        logging.info(f"DataFrame head:\n{df.head().to_string()}")  # Added detailed df head logging


        summary = extract_financial_summary(df)

        table_text = dataframe_to_text_table(df)

        error_cells = get_error_cells(file_bytes)
        if error_cells:
            summary += f"\n\nNote: Errors found in cells: {', '.join(error_cells)}"

        # Extract the detailed, structured summaries for direct, precise querying
        detailed_summaries = extract_detailed_summaries(df)
        session['detailed_summaries'] = detailed_summaries  # Store in session for subsequent chat queries

        # Store general summary and table text in session
        session['summary'] = summary
        session['table'] = table_text
        # Initialize conversation history, providing the summary and initial question to the LLM
        session['conversation'] = [{"role": "user",
                                    "content": f"Here is a summary of the financial data: {summary}\nNow, please answer this question: {question}"}]

        # First, attempt to answer the question directly using the structured data
        direct_answer = query_detailed_summary(question, detailed_summaries)

        if direct_answer:
            answer = direct_answer
            logging.info(f"Direct answer found for initial question: {answer}")
        else:
            # If a direct answer isn't found, fall back to the language model
            logging.info("Direct answer not found for initial question, falling back to LLM.")
            # Refined prompt for LLM fallback to encourage textual answers if not numerical
            prompt = (
                f"You are a financial assistant. Based on the provided financial summary, "
                f"answer the following question. If the answer is a numerical value, "
                f"please provide only the number with a '$' sign and comma formatting. "
                f"Otherwise, provide a concise textual answer, explaining if necessary.\n"
                f"Financial Summary:\n{summary}\n\n"
                f"Question: {question}\nAnswer:"
            )
            result = pipe(prompt, max_new_tokens=100,
                          do_sample=False)  # Increased max_new_tokens for more descriptive answers
            answer = result[0]['generated_text'].strip()
            logging.info(f"LLM generated answer for initial question: {answer}")

        # Append the assistant's answer to the conversation history
        session['conversation'].append({"role": "assistant", "content": answer})

        return render_template(
            'chat.html',
            answer=answer,
            filename=uploaded_file.filename,
            question=question,
            data_preview=df.head().to_html(index=False),  # Display a small HTML table preview
            table_text=table_text  # Provide the full text table for context
        )

    except Exception as e:
        logging.exception("File processing error during upload.")
        return f"Error processing file: {str(e)}", 500


@app.route('/chat', methods=['POST'])
def chat():
    question = request.form.get('question')
    if not question:
        return "No question provided", 400

    # Retrieve necessary data from the session
    summary = session.get('summary')
    table_text = session.get('table')
    conversation = session.get('conversation', [])
    detailed_summaries = session.get('detailed_summaries', {})
    session_id = session.get('session_id')

    # Validate session data to prevent errors if session has expired or data is missing
    if not summary or not detailed_summaries or not conversation or session_id not in uploaded_files:
        logging.warning(f"Session data missing for session_id: {session_id}. Please re-upload.")
        return "Session expired or missing data. Please re-upload the file.", 400

    try:
        # First, attempt to answer the question directly using the structured detailed summaries
        direct_answer = query_detailed_summary(question, detailed_summaries)

        if direct_answer:
            answer = direct_answer
            logging.info(f"Direct answer found in chat: {answer}")
        else:
            # If no direct answer, append the user's new question to the conversation history
            conversation.append({"role": "user", "content": question})

            # Format the entire conversation history for the LLM prompt
            turns = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation)

            # Construct a comprehensive prompt for the LLM, including summary and history
            prompt = (
                f"You are a financial assistant. Based on the provided financial summary and conversation history, "
                f"answer the following question. If the answer is a numerical value, "
                f"please provide only the number with a '$' sign and comma formatting. "
                f"Otherwise, provide a concise textual answer, explaining if necessary.\n\n"
                f"Financial Summary:\n{summary}\n\n"
                f"Conversation History:\n{turns}\n\n"
                f"Answer:"
            )
            result = pipe(prompt, max_new_tokens=100, do_sample=False)  # Increased max_new_tokens
            answer = result[0]['generated_text'].strip()
            logging.info(f"LLM generated answer in chat: {answer}")

            # Append the LLM's answer to the conversation history
            conversation.append({"role": "assistant", "content": answer})

        # Update the conversation history in the session
        session['conversation'] = conversation

        return render_template(
            'chat.html',
            answer=answer,
            question=question,
            table_text=table_text
        )

    except Exception as e:
        logging.exception("Chat processing error.")
        return f"Error during chat processing: {str(e)}", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

