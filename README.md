# ModelMind

This Flask application serves as an intelligent financial data assistant, allowing users to upload Excel files containing financial data and then ask questions about that data. It leverages a pre-trained language model (FLAN-T5-Small) for natural language understanding and generation, along with a custom data extraction and query engine for accurate financial calculations.

### Why it matters: 

ModelMind helps financial analysts and business users quickly gain insights from complex spreadsheets without manual number crunching — saving time, reducing errors, and making financial data more accessible through natural language questions.

### Features:
- Excel File Upload: Easily upload your financial data in .xlsx format.
- Automated Data Extraction: Automatically extracts key financial figures like Revenue, Cost of Goods Sold, Gross Profit, Operating Expenses, Operating Profit, and Net Income.
- Question Answering: Uses Flan-T5-Small model for answering questions based on the parsed document.

---

### Technologies Used:
- Flask
- Pandas
- Transformers
- spaCy
- Openpyxl
- Difflib
- HTML
- CSS (Tailwind)
- JavaScript

---

### Optimized File
You can download the file used [here](https://github.com/MilanSuri/ModelMind/blob/main/Docubridge%20Internship%20Sample%20Sheet%201.xlsx).

### Questions for LLM
- What is the total revenue for the company?
- How much revenue did the R&D department generate?
- What is the total cost of goods sold (COGS) for the company?
- Which department had the highest gross profit?
- What are the operating expenses for the Admin department?
- What is the operating profit for the Services department?
- What is the company’s total operating profit?
- What is the net income for the company after interest and taxes?
- How is gross profit calculated for each department?
- Which department had a negative operating profit, and what does that mean?
