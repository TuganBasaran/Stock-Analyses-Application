# Text-to-SQL Converter

A natural language to SQL query converter using Google's Gemini AI model, with additional RAG (Retrieval-Augmented Generation) capabilities for answering company-specific questions.

## Project Overview

This project allows users to:
- Convert natural language questions into SQL queries for a database of orders
- Query company information using RAG techniques from documentation
- Automatically detect whether a question requires SQL or company information
- Execute generated SQL queries directly on a MySQL database
- Test and evaluate the accuracy of SQL generation

## Features

- **Natural Language to SQL**: Convert plain English questions into valid MySQL queries
- **RAG Support**: Answer company-specific questions using document retrieval
- **Query Classification**: Automatically determine if a question needs SQL or RAG
- **Direct Query Execution**: Run generated SQL on a connected database
- **Accuracy Testing**: Evaluate SQL generation with test benchmarks

## Prerequisites

- Python 3.x
- MySQL database
- Google Gemini API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/TuganBasaran/AI.390-Text-to-SQL.git
   cd Text-to-SQL
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a .env file in the project root and add your Gemini API key:
   ```
   API_KEY=your_gemini_api_key
   ```

4. Create a MySQL database named `AI_390` and update database credentials in db.py if necessary.

5. Initialize the database with sample data:
   ```
   python db/run_once.py
   ```

## Project Structure

```
├── data/
│   ├── orders.csv               # Sample order data
│   ├── testbench.json           # Test cases for SQL generator  
│   └── NovaCart_RAG_Company_Info.docx  # Company info for RAG
├── db/
│   ├── db.py                    # Database connection and query execution
│   └── run_once.py              # Database initialization script
├── gemini_utils.py              # Gemini API integration
├── main_RAG.py                  # Main application for RAG and SQL queries
├── test_runner.py               # Testing framework for SQL generation
└── requirements.txt             # Project dependencies
```

## Usage

### Interactive Mode

Run the main application to interactively ask questions:

```
python main_RAG.py
```

You can then:
- Ask SQL-related questions like "What is the total revenue?"
- Ask company-related questions like "Where is the company located?"
- Type 'exit' to quit

### Testing SQL Generation

Run the test suite to evaluate the accuracy of SQL generation:

```
python test_runner.py
```

This will run through the test cases in testbench.json and show the match percentage between expected and generated SQL queries.

## Example Queries

### SQL Queries:
- "How many orders were placed in January?"
- "What is the total revenue?"
- "Which customer spent the most?"
- "List all orders placed in 2024."
- "Which month had the highest total revenue?"

### RAG Queries:
- "What is the name of the company?"
- "Where is the company located?"
- "What is the company's industry?"
- "Is customer data shared with external parties?"

## Database Structure

The project uses a single `orders` table with the following columns:
- `order_id` (Integer, Primary Key)
- `customer_name` (String)
- `order_date` (DateTime)
- `amount` (Float)

## Technologies Used

- **Google Gemini AI**: For natural language understanding and SQL generation
- **SQLAlchemy**: For database operations
- **pandas**: For data manipulation
- **python-docx**: For parsing company documentation
- **Streamlit**: For potential web interface (included in requirements)

## Contributors

- Tugan Basaran
- Zehra Sagin
