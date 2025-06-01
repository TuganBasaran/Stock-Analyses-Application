import os
import re
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys
import signal
from contextlib import contextmanager
from docx import Document
import google.generativeai as genai
from dotenv import load_dotenv
from io import StringIO
import json
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

# Import enhanced prediction engine
try:
    from enhanced_prediction_engine import EnhancedPredictionEngine
    print("✅ Enhanced prediction engine imported successfully")
except ImportError as e:
    print(f"⚠️ Enhanced prediction engine import failed: {e}")

print("✅ All required modules imported successfully")

# Timeout işleyicisi ekleyelim
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """
    Belirli bir saniye içinde çalışma zorunluluğu getiren context manager
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Code execution timed out!")
    
    # Zaman sınırını ayarla
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Reset the alarm
        signal.alarm(0)

# Import database module conditionally
try:
    import db.db as database 
    db_available = True
except Exception as e:
    db_available = False
    print(f"⚠️ Database connection error: {str(e)}")
    print("SQL features will use Pandas instead.")

# Load JSON data
try:
    # Define JSON file paths
    orders_json_path = Path("data/json/orders.json")
    products_json_path = Path("data/json/products.json")
    
    # Load orders data from JSON
    orders_data = []
    with open(orders_json_path, 'r') as file:
        for line in file:
            try:
                order = json.loads(line)
                orders_data.append({
                    'order_id': order.get('order_id'),
                    'customer_name': order.get('customer_id'),  # Using customer_id as customer_name for compatibility
                    'order_date': order.get('order_date'),
                    'amount': float(order.get('price', 0)),  # Using price as amount for compatibility
                    'location': order.get('location'), 
                    'product_id': order.get('product_id'),
                    'seller_id': order.get('seller_id', ''),  # Adding seller_id field with default empty string
                    'status': order.get('status')
                })
            except json.JSONDecodeError:
                continue
    
    # Convert to DataFrame
    csv_data = pd.DataFrame(orders_data)
    
    # Load products data from JSON
    products_data = []
    with open(products_json_path, 'r') as file:
        for line in file:
            try:
                product = json.loads(line)
                products_data.append({
                    'product_id': product.get('product_id'),
                    'category_name': product.get('category_name'),
                    'brand_name': product.get('brand_name'),
                    'release_date': product.get('release_date')
                })
            except json.JSONDecodeError:
                continue
    
    # Convert to DataFrame
    products_df = pd.DataFrame(products_data)
    
    # Merge orders with products to get category information
    csv_data = csv_data.merge(products_df, on='product_id', how='left')
    
    # Convert dates to datetime
    if 'order_date' in csv_data.columns:
        csv_data['order_date'] = pd.to_datetime(csv_data['order_date'])
    csv_available = True
    print(f"✅ Successfully loaded {len(csv_data)} orders from JSON data")
    
except Exception as e:
    csv_available = False
    print(f"⚠️ JSON files could not be loaded: {str(e)}")

# Setup API
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in .env file")
    
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

# Simple conversation history
conversation_history = []

# Function to get the next available query result file name
def get_next_query_result_filename():
    """
    No longer generates PNG filenames.
    Returns a placeholder string that will not be used for file saving.
    """
    return "visualization_placeholder"  # No actual file will be created

# RAG Konuşma geçmişini temizlemek için fonksiyon
def clear_conversation_history():
    """RAG konuşma geçmişini temizler"""
    global conversation_history
    conversation_history = []
    return len(conversation_history)

def load_company_info():
    """Loads the company information document into a single string"""
    try:
        doc_path = "data/NovaCart_RAG_Company_Info.docx"
        doc = Document(doc_path)
        # Extract all paragraphs with headings preserved
        company_text = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                # Check if it's a heading based on the style
                if para.style.name.startswith('Heading'):
                    company_text.append(f"## {text} ##")
                else:
                    company_text.append(text)
        
        # Join all paragraphs with double newlines for better readability
        full_text = "\n\n".join(company_text)
        print(f"Successfully loaded company document with {len(company_text)} paragraphs")
        return full_text
    except Exception as e:
        print(f"Error loading company information: {str(e)}")
        return None

def classify_query(query):
    """Determines if a query is about company info (RAG) or data analysis (SQL)"""
    prompt = f"""
    Classify this query as either 'SQL' or 'RAG':
    
    Query: "{query}"
    
    'SQL' = Query about data analysis, metrics, orders, customers, sales, revenue, etc.
    'RAG' = Query about company information, executives, history, products, services, etc.
    
    Return only one word: SQL or RAG
    """
    
    try:
        response = model.generate_content(prompt)
        result = response.text.strip().upper()
        if "SQL" in result:
            return "SQL"
        else:
            return "RAG"
    except Exception as e:
        print(f"Error classifying query: {str(e)}")
        # Default to RAG if classification fails
        return "RAG"

def answer_company_query(query, company_info):
    """Answers a query about company information using RAG"""
    global conversation_history
    
    # Konuşma geçmişinin kullanılması
    conversation_context = ""
    if conversation_history:
        conversation_context = "Previous conversation:\n"
        for i, (q, a) in enumerate(conversation_history):
            conversation_context += f"Question {i+1}: {q}\nAnswer {i+1}: {a}\n\n"
    
    prompt = f"""
    You are a helpful assistant for NovaCart Technologies Inc. Your task is to answer questions about the company
    based ONLY on the provided company information document. Be detailed, accurate, and professional.
    
    When responding to questions:
    1. Always look through the ENTIRE document for relevant information before responding
    2. Do NOT say "information not available" unless you've checked the entire document
    3. Use specific details and facts from the document
    4. If asked about people, search for their names and titles (especially CEO, CTO, etc.)
    5. If asked about company products, services, or history, provide comprehensive answers
    6. Provide full paragraphs, not just short phrases
    7. If the information is not in the document, explain what you were looking for and that it's not present
    8. Never make up information not found in the document
    9. Always be polite and professional in your responses
    10. If the question is not clear, ask for clarification
    11. Pay attention to the previous conversation context to understand pronouns and context-dependent questions
    
    The document contains comprehensive information about NovaCart including its leadership, 
    history, products, services and key facts. This information should be sufficient to 
    answer most questions about the company.

    {conversation_context}
    
    COMPANY INFORMATION DOCUMENT:
    ```
    {company_info}
    ```

    QUESTION: {query}

    Please provide a comprehensive answer based on the company information document.
    If the information is truly not in the document, explain what information you were looking for
    and that it's not present. Never make up information not found in the document.
    """
    
    try:
        # Use a single prompt instead of system/user roles
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,  # Lower temperature for more factual responses
                "max_output_tokens": 1024,
                "top_p": 0.95,
                "top_k": 40,
            }
        )
        answer = response.text.strip()
        
        # Konuşma geçmişine ekle (maksimum 5 soru-cevap kaydı)
        conversation_history.append((query, answer))
            
        return answer
    except Exception as e:
        print(f"Error generating company answer: {str(e)}")
        # More helpful error message with debugging info
        return f"I'm sorry, I encountered an error when trying to access company information. Technical details: {str(e)}"

def generate_sql_query(query):
    """Generates an SQL query for the orders table based on the natural language query"""
    # Column info for clarity - updated to match db.py definitions
    table_info = """
    Database tables and columns:
    
    1. Orders table columns:
       - id (INT): Auto-increment primary key
       - customer_id (VARCHAR): Customer ID
       - location (VARCHAR): Location of the order
       - order_date (DATETIME): Date when order was placed (format: YYYY-MM-DD HH:MM:SS)
       - order_id (VARCHAR): Order identifier (not a primary key)
       - price (FLOAT): Order amount in dollars
       - product_id (VARCHAR): ID of the ordered product
       - seller_id (VARCHAR): ID of the seller
       - status (ENUM): Status of the order ('Created' or 'Cancelled')
    
    2. Products table columns:
       - brand_name (VARCHAR): Brand name of the product
       - category_name (VARCHAR): Category of the product
       - product_id (VARCHAR): Primary key, matches product_id in Orders table
       - release_date (INT): Release year of the product
    """
    
    # Dinamik olarak mevcut yılı belirle
    
    current_year = datetime.now().year
    
    # Önce query içinde yıl olup olmadığını kontrol eden ön işleme aşaması
    query_preprocessor = f"""
    Analyze this query: "{query}"
    
    Does this query mention a specific month without specifying a year? For example, queries like:
    - "How many orders were placed in January?"
    - "Show me sales from March"
    - "What was the revenue in December?"
    
    If a month is mentioned without a year, we should assume the current year ({current_year}).
    Answer with just YES or NO.
    """
    
    try:
        # Ön işleme için model yanıtını al
        preprocess_response = model.generate_content(
            query_preprocessor,
            generation_config={"temperature": 0.1, "max_output_tokens": 10}
        )
        needs_year = "YES" in preprocess_response.text.strip().upper()
        
        # Ana sorgu için prompt hazırla
        prompt = f"""
        Generate a MySQL SQL query for the following question about order and product data:
        
        Question: {query}
        
        {table_info}
        
        Today's date is {datetime.now().strftime('%B %d, %Y')}.
        
        Important guidelines:
        - Return ONLY the SQL query without any explanations
        - Use MySQL syntax
        - For month filtering, use MONTH() function, e.g., WHERE MONTH(order_date) = 1 for January
        - For year filtering, use YEAR() function, e.g., WHERE YEAR(order_date) = 2021
        - For quarterly analysis, use MONTH() and CASE function: 
          CASE 
            WHEN MONTH(order_date) IN (1,2,3) THEN 1
            WHEN MONTH(order_date) IN (4,5,6) THEN 2
            WHEN MONTH(order_date) IN (7,8,9) THEN 3
            WHEN MONTH(order_date) IN (10,11,12) THEN 4
          END as quarter
        - For seasonal analysis, use CASE statements with MONTH() to group months into seasons:
          CASE 
            WHEN MONTH(order_date) IN (12, 1, 2) THEN 'Winter'
            WHEN MONTH(order_date) IN (3, 4, 5) THEN 'Spring'
            WHEN MONTH(order_date) IN (6, 7, 8) THEN 'Summer'
            WHEN MONTH(order_date) IN (9, 10, 11) THEN 'Fall'
          END as season
        - Table names are "Orders" and "Products" (with capital first letter)
        - Use the exact column names as specified above
        - Join tables when the question requires data from both Orders and Products
        - For example, to join: "SELECT o.price, p.category_name FROM Orders o JOIN Products p ON o.product_id = p.product_id"
        - The status column is an ENUM with values 'Created' or 'Cancelled'
        - For category analysis, use p.category_name from Products table
        - For trend analysis, include aggregation functions like SUM(o.price), COUNT(*), AVG(o.price)
        - For top categories, use ORDER BY and LIMIT clauses
        - For quarterly/seasonal comparisons, use GROUP BY with time functions
        """
        
        # Eğer ay belirtilmiş ve yıl belirtilmemişse, otomatik olarak mevcut yıl filtresi ekle
        if needs_year:
            prompt += f"""
        - IMPORTANT: The query mentions a month without specifying a year. 
          Add a filter for the current year ({current_year}) like this: AND YEAR(order_date) = {current_year}
            """
        
        # SQL sorgusunu oluştur
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 1024,
                "top_p": 0.95,
                "top_k": 40,
            }
        )
        
        # Clean up the response
        sql = response.text.strip()
        # Remove code block markers if present
        sql = re.sub(r'^```sql\s*', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'\s*```$', '', sql, flags=re.MULTILINE)
        
        # Oluşturulan sorguyu son bir kez kontrol et ve gerekirse yıl filtresi ekle
        if needs_year and "year(order_date)" not in sql.lower():
            # SQL sorgusuna WHERE ya da AND ile yıl koşulu ekle
            if "where" in sql.lower():
                # WHERE zaten var, AND ile ekle
                sql = sql.replace(";", "")  # Noktalı virgülü kaldır
                sql += f" AND YEAR(order_date) = {current_year};"
            else:
                # WHERE yok, WHERE ile ekle
                sql = sql.replace(";", "")  # Noktalı virgülü kaldır
                sql += f" WHERE YEAR(order_date) = {current_year};"
        
        return sql.strip()
    except Exception as e:
        print(f"Error generating SQL query: {str(e)}")
        return "SELECT * FROM Orders LIMIT 5 -- Error generating query"

def generate_direct_pandas_code(query, result_data=None):
    """Generates pandas code directly from natural language, without SQL as intermediate step"""
    
    # Check if result_data is provided and is empty
    if result_data is not None and (not result_data or len(result_data) == 0):
        print("No data available from SQL query result. Skipping pandas code generation.")
        return None
    
    dataframe_info = """
    DataFrame name: csv_data
    Columns:
    - order_id (VARCHAR): Order identifier
    - customer_name (VARCHAR): Customer ID (same as customer_id in database)
    - order_date (DATETIME): Date when order was placed (format: YYYY-MM-DD HH:MM:SS)
    - amount (FLOAT): Order price in dollars (same as price in database)
    - location (STRING): Location where the order was placed
    - product_id (STRING): ID of the ordered product
    - seller_id (STRING): ID of the seller
    - status (STRING): Status of the order ('Created' or 'Cancelled')
    - category_name (STRING): Product category name (from Products table)
    - brand_name (STRING): Product brand name (from Products table)
    - release_date (INT): Product release year (from Products table)
    """
    
    # Get the next available filename
    result_filename = get_next_query_result_filename()

    current_year = datetime.now().year

    
    prompt = f"""
    Generate Python code using Pandas to analyze data for this question: "{query}"
    
    {dataframe_info}
    
    Important guidelines:
    - Generate ONLY runnable Python code with no explanations
    - The code must work with the pre-loaded 'csv_data' DataFrame
    - Your code should both analyze the data AND create a visualization
    - For visualization, use ONLY matplotlib (which is already imported)
    - Save the visualization to '{result_filename}' with dpi=300
    - Include code to print key insights from the analysis
    - Make sure to close matplotlib figures at the end
    - The code should handle any errors gracefully
    - Make the visualization look professional with proper labels, titles, etc.
    - If the question mentions a month without a year, assume the current year ({current_year})
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
                "top_p": 0.95,
                "top_k": 40,
            }
        )
        code = response.text.strip()
        # Remove code block markers if present
        code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'\s*```$', '', code, flags=re.MULTILINE)
        return code.strip()
    except Exception as e:
        print(f"Error generating pandas code: {str(e)}")
        return f"print('Error generating pandas code: {str(e)}')"

def execute_full_llm_query(query, sql_query=None, sql_result=None):
    """Executes the entire query process using a single LLM call for maximum flexibility
    
    This function uses the LLM to:
    1. Analyze the query intent
    2. Generate appropriate code to process the data
    3. Return analysis results WITHOUT creating a PNG file
    
    All in a single LLM call without predefined constraints
    """
    try:
        print(f"Processing query using full LLM approach: {query}")
        
        current_year = datetime.now().year
        
        # Create a detailed prompt that asks for analysis without saving visualization
        prompt = f"""
        You are an expert data analyst. You need to analyze data based on this natural language query: "{query}"
        
        Here is information about the available data:
        1. DataFrame name: csv_data
        2. DataFrame columns:
           - order_id (VARCHAR): Order identifier 
           - customer_name (VARCHAR): Customer ID
           - order_date (DATETIME): Date when order was placed (format: YYYY-MM-DD HH:MM:SS)
           - amount (FLOAT): Order price in dollars (same as 'price')
           - location (STRING): Location where the order was placed
           - product_id (STRING): ID of the ordered product
           - status (STRING): Status of the order ('Created' or 'Cancelled')
           - seller_id (STRING): ID of the seller (may be missing in some records)
           - category_name (STRING): Product category name (from Products table)
           - brand_name (STRING): Product brand name (from Products table)
           - release_date (INT): Product release year (from Products table)
        
        """
        
        # Add SQL query context if provided
        if sql_query:
            prompt += f"""
        SQL query that was executed: {sql_query}
        SQL query result summary: {sql_result}
            """
        
        prompt += f"""
        Create a complete solution that:
        1. Understands the user's intent from their query
        2. Generates appropriate Python code to analyze the data
        3. Performs data analysis and returns insights
        
        Your solution must follow these CRITICAL guidelines:
        - Return ONLY executable Python code with NO explanations or comments
        - The code should be self-contained and analyze the data
        - Use ONLY pandas, numpy, and matplotlib which are already imported
        - Use the pre-loaded DataFrame 'csv_data'
        - CREATE matplotlib visualizations but DO NOT save them to files
        - DO NOT use plt.savefig() anywhere in your code
        - DO NOT use matplotlib's savefig method anywhere
        - DO NOT create or save ANY .png, .jpg, .jpeg, .svg or ANY other image file
        - DO NOT use ANY filename like 'highest_spending.png', 'query_result.png', etc.
        - Create beautiful and informative plots using plt.figure(), plt.plot(), plt.bar(), etc.
        - Include code to print key insights from the data analysis
        - Print a summary of findings that would be helpful for the user
        - Handle potential errors gracefully
        - DO NOT close figures with plt.close() - leave them open for capture
        - Show visualizations with plt.show() instead of saving
        
        The final output should be ONLY the executable Python code that accomplishes all of the above.
        """
        
        # Get the complete solution from LLM
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 4096,
                "top_p": 0.95,
                "top_k": 40,
            }
        )
        
        # Extract the code
        full_code = response.text.strip()
        # Remove code block markers if present
        full_code = re.sub(r'^```python\s*', '', full_code, flags=re.MULTILINE)
        full_code = re.sub(r'\s*```$', '', full_code, flags=re.MULTILINE)
        
        # Replace any plt.savefig or any other file saving operations with pass statements
        full_code = re.sub(r'plt\.savefig\([^\)]+\)', '# File saving removed', full_code)
        full_code = re.sub(r'\.savefig\([^\)]+\)', '.show()', full_code)
        
        print(f"\n📝 Generated Complete Solution:\n{full_code}")
        
        # Execute the code with full freedom
        print("\nExecuting LLM solution...")
        
        # Create a local scope with csv_data
        local_vars = {"csv_data": csv_data.copy(), "pd": pd, "np": np, "plt": plt}
        
        # Capture print outputs
        original_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # Add a print statement at the beginning of the code for debugging
            debug_code = "print('Debug: Starting code execution...')\n" + full_code
            
            # Add array validation and better figure capture
            array_validation_code = """
# Add validation checks to ensure no array mismatches
import numpy as np

def validate_arrays_in_figure(fig):
    '''
    Check that all arrays in the figure have compatible lengths
    '''
    if not fig or not hasattr(fig, 'get_axes'):
        return True
    
    for ax in fig.get_axes():
        # Check all line objects
        for line in ax.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            
            if len(x_data) != len(y_data):
                print(f"WARNING: Data length mismatch detected: x={len(x_data)}, y={len(y_data)}")
                # Try to fix by truncating to common length
                min_len = min(len(x_data), len(y_data))
                line.set_data(x_data[:min_len], y_data[:min_len])
                
        # Check bar plots
        for container in ax.containers:
            if hasattr(container, 'datavalues') and hasattr(container, 'patches'):
                if len(container.datavalues) != len(container.patches):
                    print(f"WARNING: Bar data mismatch: values={len(container.datavalues)}, bars={len(container.patches)}")
    
    return True

# Capture figure before closing
try:
    current_fig = plt.gcf()
    if current_fig and hasattr(current_fig, 'get_axes') and current_fig.get_axes():
        # Validate arrays in the figure
        validate_arrays_in_figure(current_fig)
        fig_captured = current_fig
    else:
        fig_captured = None
except Exception as fig_error:
    print(f"WARNING: Figure capture error: {fig_error}")
    fig_captured = None

print('Debug: Figure capture completed...')
"""
            
            # Add the validation code to the end of the debug code
            debug_code_with_capture = debug_code + "\n\n" + array_validation_code
            
            # Execute the LLM-generated code with a debug print
            print("Debug: About to execute code...")
            with time_limit(60):  # 60 second time limit
                exec(debug_code_with_capture, {"__builtins__": __builtins__}, local_vars)
            print("Debug: Code execution completed.")
            
            # Restore stdout
            sys.stdout = original_stdout
            
            # Get the captured output
            output = captured_output.getvalue()
            print("\n📊 Analysis Results:")
            print(output)
            
            # Try to get the captured figure from local_vars
            captured_fig = local_vars.get('fig_captured', None)
            
            # Return success with the captured figure
            if captured_fig is not None:
                return True, "Analysis completed successfully with visualization.", captured_fig
            else:
                return True, "Analysis completed successfully. No visualization was created.", None
                
        except TimeoutException as timeout_error:
            # Restore stdout in case of timeout
            sys.stdout = original_stdout
            print(f"❌ Error: {str(timeout_error)}")
            return False, f"Execution timed out: {str(timeout_error)}", None
        except Exception as exec_error:
            # Restore stdout in case of error
            sys.stdout = original_stdout
            print(f"❌ Error executing LLM solution: {str(exec_error)}")
            error_line = getattr(exec_error, 'lineno', 'unknown line')
            print(f"Error occurred at line: {error_line}")
            return False, f"An error occurred: {str(exec_error)}", None
            
    except Exception as e:
        print(f"Error in full LLM execution: {str(e)}")
        return False, f"Error: {str(e)}", None

# Function to ensure all file names follow the standard naming convention
def standardize_file_name(file_name):
    """
    This function no longer creates PNG files.
    Instead it returns a placeholder name that won't be used for file saving.
    """
    # We no longer need to check or generate actual filenames
    return "visualization_placeholder"

def generate_prediction(query):
    """
    Enhanced prediction function using the new unified prediction engine.
    Optimized for reliability, speed, and better business insights.
    """
    global csv_data, model, db_available
    
    try:
        # Initialize enhanced prediction engine with the global model
        engine = EnhancedPredictionEngine(model)
        
        # Check if this is a prediction query
        prediction_keywords = ['predict', 'forecast', 'future', 'next month', 'next year', 
                              'upcoming', 'will be', 'expect', 'projected', 'anticipated']
        
        query_lower = query.lower()
        is_prediction_query = any(keyword in query_lower for keyword in prediction_keywords)
        
        if not is_prediction_query:
            return False, "This query does not appear to be requesting a prediction or forecast analysis.", None, None
        
        # Get data from SQL database if available, otherwise use JSON data
        prediction_data = None
        data_source = ""
        
        try:
            if db_available:
                print("🔗 Fetching data from SQL database...")
                import db.db as database
                
                # Get all order data from SQL database
                data_query = """
                SELECT 
                    o.order_id,
                    o.customer_id as customer_name,
                    o.order_date,
                    o.price as amount,
                    o.location,
                    o.product_id,
                    o.status,
                    o.seller_id
                FROM Orders o
                WHERE o.status != 'Cancelled'
                ORDER BY o.order_date ASC
                """
                
                # Execute query and get results
                sql_results = database.execute_raw_query(data_query)
                
                if sql_results:
                    # Convert SQL results to DataFrame
                    prediction_data = pd.DataFrame(sql_results)
                    # Ensure order_date is datetime
                    prediction_data['order_date'] = pd.to_datetime(prediction_data['order_date'])
                    data_source = "SQL Database"
                    print(f"✅ Loaded {len(prediction_data)} records from SQL database")
                else:
                    raise Exception("No data returned from SQL database")
                    
        except Exception as db_error:
            print(f"⚠️ SQL database error: {str(db_error)}")
            print("📋 Falling back to JSON data...")
            db_available = False
        
        if not db_available and csv_available:
            # Use JSON data as fallback
            prediction_data = csv_data.copy()
            # Filter out cancelled orders if status column exists
            if 'status' in prediction_data.columns:
                prediction_data = prediction_data[prediction_data['status'] != 'Cancelled']
            data_source = "JSON Files"
            print(f"✅ Using {len(prediction_data)} orders from JSON data")
        
        if prediction_data is None or len(prediction_data) == 0:
            return False, "No data source is available for prediction analysis. Please ensure your database contains historical data.", None, None
        
        print(f"🔮 Processing prediction query: {query}")
        print(f"📊 Data source: {data_source}")
        
        # Use enhanced prediction engine
        success, insights, figure, forecasts_df = engine.create_prediction(query, prediction_data)
        
        if success:
            # Add data source information to insights
            enhanced_insights = f"Data Source: {data_source}\n\n{insights}"
            print("✅ Enhanced prediction analysis completed successfully")
            return True, enhanced_insights, figure, forecasts_df
        else:
            print(f"⚠️ Enhanced prediction failed: {insights}")
            return False, insights, None, None
            
    except Exception as e:
        print(f"❌ Error in enhanced prediction: {str(e)}")
        return False, f"Error generating predictive analysis: {str(e)}", None, None

# Function to validate matplotlib figure arrays
def validate_matplotlib_arrays(fig):
    """
    Validate that all arrays in matplotlib figure have consistent lengths.
    Returns tuple: (is_valid, error_message)
    """
    try:
        if fig is None or not hasattr(fig, 'get_axes'):
            return True, ""
        
        # Check each axes and its elements
        for ax_idx, ax in enumerate(fig.get_axes()):
            # Check lines (most common source of array mismatch errors)
            for line_idx, line in enumerate(ax.get_lines()):
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                
                if len(x_data) != len(y_data):
                    error_msg = f"Array length mismatch in axis {ax_idx}, line {line_idx}: x={len(x_data)}, y={len(y_data)}"
                    print(f"WARNING: {error_msg}")
                    return False, error_msg
            
            # Check collections (patches, bar plots, etc.)
            for col_idx, collection in enumerate(ax.collections):
                if hasattr(collection, '_offsets') and collection._offsets is not None:
                    if hasattr(collection, '_sizes') and collection._sizes is not None:
                        if len(collection._offsets) != len(collection._sizes):
                            error_msg = f"Collection size mismatch in axis {ax_idx}, collection {col_idx}"
                            print(f"WARNING: {error_msg}")
                            return False, error_msg
        
        return True, ""
    except Exception as e:
        return False, f"Error validating figure arrays: {str(e)}"

# Main execution loop
if __name__ == "__main__":
    print("\n🌟 Welcome to AI-Powered Text-to-SQL System! 🌟")
    print("You can ask questions about the company or analyze order data.")
    print("Examples:")
    print(" - 'Who is the CEO of NovaCart?'")
    print(" - 'How many orders were placed in January?'")
    print(" - 'What is the total revenue?'")
    print("\nPandas Specific Examples:")
    print(" - 'Show total sales by customer using pandas'")
    print(" - 'Calculate monthly revenue trends with pandas'")
    print(" - 'Find top 3 customers by order value with pandas'")
    print("Type 'exit' to quit.")
    
    # Load company information for RAG
    print("\nLoading company information...")
    company_info = load_company_info()
    if company_info:
        print("✅ Company information loaded successfully!")
    else:
        print("⚠️ Could not load company information. Company queries may not work.")
    
    # Check database and CSV status
    if not db_available:
        print("⚠️ MySQL database not available. Using Pandas for data processing.")
    if not csv_available:
        print("⚠️ CSV data could not be loaded. Data queries will not work.")
    
    # Check for pandas force mode from command arguments
    import sys
    force_pandas = "--pandas" in sys.argv
    if force_pandas:
        print("🐼 Pandas mode forced: All SQL queries will be processed with Pandas")
        db_available = False
    
    # Main interaction loop
    while True:
        # Get user query
        query = input("\n🔍 Ask a question (or type 'exit', 'history', 'clear history', 'clear chat history'): ")
        
        # Check for exit command
        if query.lower() == "exit":
            print("\nThank you for using the AI-Powered Text-to-SQL System. Goodbye!")
            break
        
        # Check for history command
        if query.lower() == "history":
            if db_available:
                print("\n📜 Query History:")
                database.get_query_history()
            else:
                print("⚠️ Cannot show query history: database not available.")
            continue
            
        # Check for clear history command
        if query.lower() == "clear history":
            if db_available:
                deleted_count = database.clear_query_history()
                print(f"\n🧹 Query history cleared. {deleted_count} records deleted.")
            else:
                print("⚠️ Cannot clear query history: database not available.")
            continue
            
        # Check for clear chat history command
        if query.lower() == "clear chat history":
            deleted_count = clear_conversation_history()
            print(f"\n🧹 Chat conversation history cleared.")
            continue
            
        # Skip empty queries
        if not query.strip():
            continue
            
        try:
            # Classify the query type
            query_type = classify_query(query)
            print(f"\nQuery classified as: {query_type}")
            
            # Process based on query type
            if query_type == "RAG":
                # Company information query
                if company_info:
                    print("Retrieving company information...")
                    answer = answer_company_query(query, company_info)
                    print(f"\n🏢 Company Information:\n{answer}")
                else:
                    print("⚠️ Cannot answer company questions: company information not loaded.")
                    
            else:  # SQL query
                if not csv_available and not db_available:
                    print("⚠️ Cannot process data queries: neither database nor CSV data available.")
                    continue
                
                # Generate SQL query
                sql_query = generate_sql_query(query)
                print(f"\n📊 Generated SQL Query:\n{sql_query}")
                
                # Execute query
                print("\nExecuting query...")
                
                if db_available:
                    # Try with database first
                    try:
                        print("\n📋 Database Results:")
                        # Kullanıcı girdisini de history'ye eklemek için güncellenmiş call
                        database.execute_query(sql_query, query)
                        print("✅ Query executed successfully!")
                        
                        # Even if database query is successful, use full LLM approach for visualization
                        if csv_available:
                            # Try the full LLM approach for maximum flexibility
                            print("\nGenerating complete analysis...")
                            success, message, _ = execute_full_llm_query(query, sql_query)
                            if success:
                                print(f"📈 {message}")
                            else:
                                print(f"⚠️ {message}")
                    except Exception as e:
                        print(f"⚠️ Database query failed: {str(e)}")
                        print("Falling back to full LLM solution...")
                        if csv_available:
                            # Use full LLM approach
                            success, message, _ = execute_full_llm_query(query, sql_query)
                            if success:
                                print(f"📈 {message}")
                            else:
                                print(f"⚠️ {message}")
                else:
                    # Use full LLM approach directly
                    if csv_available:
                        # Use full LLM approach
                        success, message, _ = execute_full_llm_query(query)
                        if success:
                            print(f"📈 {message}")
                        else:
                            print(f"⚠️ {message}")
                    else:
                        print("⚠️ Cannot execute query: CSV data not available.")
            
        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}")
            print("Please try asking in a different way.")
        
        # Visual separator between interactions
        print("\n" + "-" * 50)

def generate_simple_prediction(query, prediction_data, data_source="Data"):
    """
    Prompt-based prediction function that uses AI to generate customized analysis code 
    based on the specific query, allowing maximum flexibility without predefined functions.
    """
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        from io import StringIO
        import sys
        
        print(f"🤖 Generating AI-powered prediction for: {query}")
        print(f"📋 Data source: {data_source}")
        print(f"📈 Records available: {len(prediction_data)}")
        
        # Prepare the data
        data = prediction_data.copy()
        data['order_date'] = pd.to_datetime(data['order_date'])
        
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Create a comprehensive prompt for AI-generated prediction code
        prompt = f'''Generate Python code for business forecasting and prediction analysis based on this query: "{query}"

CRITICAL: Generate DIRECT EXECUTION CODE, not functions. Code must execute immediately.

Data available:
- DataFrame: data with columns: order_date (datetime), amount (float), status, location, etc.
- Current date: {current_year}-{current_month:02d}
- Data source: {data_source}

REQUIRED CODE STRUCTURE (execute directly, no functions):
```python
# 1. Data preparation and aggregation
monthly_data = data.groupby(pd.Grouper(key='order_date', freq='ME'))['amount'].sum()
monthly_data = monthly_data[monthly_data > 0]  # Remove zero months

# 2. Generate appropriate analysis based on query type
# For revenue prediction: simple moving average and trend
# For growth rate: calculate month-over-month growth percentages
# For trends: multi-metric analysis with volume and value
# For quarterly: quarter aggregation and QoQ growth
# For 6-month: multiple forecast methods with ensemble

# 3. Create professional visualization with matplotlib
plt.figure(figsize=(14, 8))
# Add multiple subplots if needed for comprehensive analysis
# Include historical data, trends, and forecasts

# 4. Print detailed analysis results
print("📊 PREDICTION ANALYSIS RESULTS")
print("=" * 40)
print(f"Query: {query}")
print(f"Data Source: {data_source}")

# 5. Store forecast results
forecasts_df = pd.DataFrame()  # Must create this for return value
```

CRITICAL REQUIREMENTS:
- DO NOT create functions or classes
- DO NOT create mock/fake data - use the 'data' DataFrame
- Code must execute line by line immediately
- Use ONLY pandas, numpy, matplotlib (already imported)
- Include comprehensive error handling with try/except
- Generate meaningful business insights and numerical forecasts
- Create professional visualizations with titles, labels, grids
- Print detailed analysis results with business recommendations
- Store forecast results in forecasts_df DataFrame
- Generate ONLY executable Python code, no explanations or markdown'''

        try:
            # Generate prediction code using AI
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,  # Lower temperature for more consistent code
                    "max_output_tokens": 3072,
                    "top_p": 0.9,
                    "top_k": 40,
                }
            )
            
            # Extract and clean the generated code
            prediction_code = response.text.strip()
            prediction_code = re.sub(r'^```python\s*', '', prediction_code, flags=re.MULTILINE)
            prediction_code = re.sub(r'\s*```$', '', prediction_code, flags=re.MULTILINE)
            
            print(f"\n🤖 Generated Prediction Code:\n{prediction_code}\n")
            
            # Capture output from code execution
            original_stdout = sys.stdout
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                # Create execution environment with required variables
                exec_globals = {
                    '__builtins__': __builtins__,
                    'data': data.copy(),
                    'pd': pd,
                    'np': np, 
                    'plt': plt,
                    'datetime': datetime,
                    'timedelta': timedelta,
                    're': re,
                    'StringIO': StringIO,
                    'sys': sys,
                    'forecasts_df': pd.DataFrame(),  # Initialize forecasts_df
                    'data_source': data_source,
                    'query': query
                }
                
                # Execute the AI-generated prediction code
                print("🔄 Executing AI-generated prediction analysis...")
                exec(prediction_code, exec_globals)
                
                # Restore stdout and get output
                sys.stdout = original_stdout
                output = captured_output.getvalue()
                
                print(f"\n📊 Prediction Analysis Results ({data_source}):")
                print(output)
                
                # Get the forecasts_df from execution context
                forecasts_df = exec_globals.get('forecasts_df', pd.DataFrame())
                
                # Try to get the current figure if one was created
                try:
                    current_fig = plt.gcf()
                    if current_fig.get_axes():  # Check if figure has any plots
                        current_fig.tight_layout()
                        print("✅ AI-generated prediction completed successfully")
                        return True, output, current_fig, forecasts_df
                    else:
                        print("⚠️ No visualization was generated")
                        return True, output, None, forecasts_df
                except Exception as fig_error:
                    print(f"⚠️ Warning: Error retrieving matplotlib figure: {fig_error}")
                    return True, output + f"\n\nNote: Visualization error occurred: {str(fig_error)}", None, forecasts_df
                    
            except Exception as exec_error:
                sys.stdout = original_stdout
                print(f"❌ Error executing AI-generated prediction code: {str(exec_error)}")
                
                # Fallback to basic monthly revenue prediction
                try:
                    return _predict_monthly_revenue(data, query, data_source)
                except Exception as fallback_error:
                    print(f"❌ Fallback prediction also failed: {str(fallback_error)}")
                    return False, f"Error in prediction analysis execution: {str(exec_error)}", None, None
                    
        except Exception as ai_error:
            print(f"❌ Error generating AI prediction code: {str(ai_error)}")
            # Final fallback to basic prediction
            try:
                return _predict_monthly_revenue(data, query, data_source)
            except Exception as final_error:
                print(f"❌ All prediction methods failed: {str(final_error)}")
                return False, f"Error generating prediction analysis: {str(ai_error)}", None, None
            
    except Exception as e:
        error_msg = f"Error in prediction analysis: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg, None, None

def _predict_monthly_revenue(data, query, data_source):
    """Predict next month's revenue"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    try:
        # Create monthly aggregation
        monthly_revenue = data.groupby(pd.Grouper(key='order_date', freq='ME'))['amount'].sum()
        monthly_revenue = monthly_revenue[monthly_revenue > 0]  # Remove zero months
        
        if len(monthly_revenue) < 3:
            return False, "Insufficient historical data for prediction (need at least 3 months)", None, None
        
        print(f"📅 Historical data: {len(monthly_revenue)} months")
        print(f"💰 Latest month revenue: ${monthly_revenue.iloc[-1]:,.2f}")
        
        # Simple forecasting methods
        # 1. Moving average of last 3 months
        ma_forecast = monthly_revenue.tail(3).mean()
        
        # 2. Linear trend
        x = np.arange(len(monthly_revenue))
        y = monthly_revenue.values
        slope, intercept = np.polyfit(x, y, 1)
        trend_forecast = slope * len(monthly_revenue) + intercept
        
        # 3. Ensemble (average of methods)
        ensemble_forecast = (ma_forecast + trend_forecast) / 2
        
        # Next month date
        next_month = monthly_revenue.index[-1] + pd.DateOffset(months=1)
        
        # Create forecast DataFrame
        forecasts_df = pd.DataFrame({
            'date': [next_month],
            'moving_average': [ma_forecast],
            'linear_trend': [trend_forecast]
        })
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(monthly_revenue.index, monthly_revenue.values, 
                marker='o', linewidth=2, label='Historical Revenue', color='blue')
        
        # Plot forecasts
        plt.scatter(next_month, ma_forecast, color='green', s=100, marker='s', 
                   label=f'Moving Average: ${ma_forecast:,.0f}')
        plt.scatter(next_month, trend_forecast, color='orange', s=100, marker='^', 
                   label=f'Linear Trend: ${trend_forecast:,.0f}')
        plt.scatter(next_month, ensemble_forecast, color='red', s=150, marker='*', 
                   label=f'Ensemble: ${ensemble_forecast:,.0f}')
        
        # Formatting
        plt.title('Monthly Revenue Forecast', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Revenue ($)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add data source annotation
        plt.text(0.02, 0.98, f'Data Source: {data_source}', 
                transform=plt.gca().transAxes, fontsize=8, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Get the figure
        fig = plt.gcf()
        
        # Create analysis report
        growth_rate = ((monthly_revenue.iloc[-1] / monthly_revenue.iloc[0]) ** (1/(len(monthly_revenue)-1)) - 1) * 100
        
        report = f"""
📊 PREDICTION ANALYSIS REPORT
══════════════════════════════

🎯 Query: {query}
📅 Forecast Period: {next_month.strftime('%B %Y')}
📈 Historical Period: {monthly_revenue.index[0].strftime('%B %Y')} to {monthly_revenue.index[-1].strftime('%B %Y')}

💰 FORECAST RESULTS:
• Moving Average (3-month): ${ma_forecast:,.2f}
• Linear Trend: ${trend_forecast:,.2f}
• Ensemble Forecast: ${ensemble_forecast:,.2f}

📊 HISTORICAL INSIGHTS:
• Total months analyzed: {len(monthly_revenue)}
• Average monthly revenue: ${monthly_revenue.mean():,.2f}
• Latest month revenue: ${monthly_revenue.iloc[-1]:,.2f}
• Monthly growth rate: {growth_rate:+.1f}%
• Revenue range: ${monthly_revenue.min():,.2f} - ${monthly_revenue.max():,.2f}

🔮 BUSINESS RECOMMENDATIONS:
• The ensemble forecast suggests {next_month.strftime('%B %Y')} revenue will be approximately ${ensemble_forecast:,.2f}
• This represents a {"growth" if ensemble_forecast > monthly_revenue.iloc[-1] else "decline"} from the previous month
• Consider seasonal factors and external market conditions for more accurate planning
• Monitor actual results against predictions to improve future forecasting accuracy

Data Source: {data_source}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        print(report)
        return True, report.strip(), fig, forecasts_df
        
    except Exception as e:
        print(f"❌ Error in simple prediction: {str(e)}")
        return False, f"Error generating prediction: {str(e)}", None, None

def _predict_growth_rate(data, query, data_source):
    """Predict growth rate analysis"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    try:
        # Calculate monthly revenue
        monthly_revenue = data.groupby(pd.Grouper(key='order_date', freq='ME'))['amount'].sum()
        monthly_revenue = monthly_revenue[monthly_revenue > 0]
        
        if len(monthly_revenue) < 3:
            return False, "Insufficient data for growth rate analysis", None, None
        
        # Calculate month-over-month growth rates
        growth_rates = monthly_revenue.pct_change() * 100
        growth_rates = growth_rates.dropna()
        
        # Calculate various growth metrics
        avg_growth = growth_rates.mean()
        recent_growth = growth_rates.tail(3).mean()
        annual_growth = ((monthly_revenue.iloc[-1] / monthly_revenue.iloc[0]) ** (12/len(monthly_revenue)) - 1) * 100
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Plot 1: Revenue over time
        plt.subplot(2, 2, 1)
        plt.plot(monthly_revenue.index, monthly_revenue.values, marker='o', linewidth=2, color='blue')
        plt.title('Monthly Revenue Trend', fontweight='bold')
        plt.ylabel('Revenue ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Growth rates
        plt.subplot(2, 2, 2)
        plt.bar(growth_rates.index, growth_rates.values, alpha=0.7, color='green')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        plt.title('Month-over-Month Growth Rate', fontweight='bold')
        plt.ylabel('Growth Rate (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative growth
        plt.subplot(2, 2, 3)
        cumulative_growth = ((monthly_revenue / monthly_revenue.iloc[0] - 1) * 100)
        plt.plot(cumulative_growth.index, cumulative_growth.values, marker='s', linewidth=2, color='purple')
        plt.title('Cumulative Growth from Start', fontweight='bold')
        plt.ylabel('Cumulative Growth (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Growth rate distribution
        plt.subplot(2, 2, 4)
        plt.hist(growth_rates.values, bins=10, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(x=avg_growth, color='red', linestyle='--', label=f'Average: {avg_growth:.1f}%')
        plt.title('Growth Rate Distribution', fontweight='bold')
        plt.xlabel('Growth Rate (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig = plt.gcf()
        
        # Create analysis report
        report = f"""
📈 GROWTH RATE ANALYSIS REPORT
═══════════════════════════════

🎯 Query: {query}
📅 Analysis Period: {monthly_revenue.index[0].strftime('%B %Y')} to {monthly_revenue.index[-1].strftime('%B %Y')}

📊 GROWTH METRICS:
• Average Monthly Growth: {avg_growth:+.2f}%
• Recent 3-Month Growth: {recent_growth:+.2f}%
• Annualized Growth Rate: {annual_growth:+.2f}%
• Best Month Growth: {growth_rates.max():+.2f}%
• Worst Month Growth: {growth_rates.min():+.2f}%

📈 PERFORMANCE INSIGHTS:
• Total months analyzed: {len(monthly_revenue)}
• Positive growth months: {(growth_rates > 0).sum()}/{len(growth_rates)}
• Growth volatility (std dev): {growth_rates.std():.2f}%
• Revenue multiplier: {monthly_revenue.iloc[-1]/monthly_revenue.iloc[0]:.2f}x

🔮 GROWTH OUTLOOK:
• Based on recent trends, {"strong positive" if recent_growth > 5 else "moderate positive" if recent_growth > 0 else "negative"} growth trajectory
• Growth consistency: {"High" if growth_rates.std() < 10 else "Medium" if growth_rates.std() < 20 else "Low"}
• Recommended focus: {"Sustain momentum" if recent_growth > avg_growth else "Improve growth strategies"}

Data Source: {data_source}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return True, report.strip(), fig, None
        
    except Exception as e:
        return False, f"Error in growth rate analysis: {str(e)}", None, None

def _predict_quarterly_trends(data, query, data_source):
    """Predict quarterly trends analysis"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    try:
        # Calculate quarterly revenue
        quarterly_revenue = data.groupby(pd.Grouper(key='order_date', freq='Q'))['amount'].sum()
        quarterly_revenue = quarterly_revenue[quarterly_revenue > 0]
        
        if len(quarterly_revenue) < 2:
            return False, "Insufficient data for quarterly analysis", None, None
        
        # Calculate quarter-over-quarter growth
        qoq_growth = quarterly_revenue.pct_change() * 100
        qoq_growth = qoq_growth.dropna()
        
        # Forecast next quarter using trend
        if len(quarterly_revenue) >= 3:
            x = np.arange(len(quarterly_revenue))
            y = quarterly_revenue.values
            slope, intercept = np.polyfit(x, y, 1)
            next_quarter_forecast = slope * len(quarterly_revenue) + intercept
        else:
            next_quarter_forecast = quarterly_revenue.iloc[-1] * (1 + qoq_growth.mean()/100)
        
        # Create visualization
        plt.figure(figsize=(14, 6))
        
        # Plot quarterly revenue
        plt.subplot(1, 2, 1)
        plt.bar(range(len(quarterly_revenue)), quarterly_revenue.values, alpha=0.8, color='skyblue', edgecolor='navy')
        plt.plot(range(len(quarterly_revenue)), quarterly_revenue.values, marker='o', linewidth=2, color='red')
        
        # Add forecast
        plt.bar(len(quarterly_revenue), next_quarter_forecast, alpha=0.8, color='lightcoral', edgecolor='darkred', label='Forecast')
        
        quarter_labels = [f"Q{q.quarter}\n{q.year}" for q in quarterly_revenue.index]
        quarter_labels.append(f"Q{(quarterly_revenue.index[-1] + pd.DateOffset(months=3)).quarter}\n{(quarterly_revenue.index[-1] + pd.DateOffset(months=3)).year}")
        
        plt.xticks(range(len(quarter_labels)), quarter_labels, rotation=45)
        plt.title('Quarterly Revenue Trends', fontweight='bold')
        plt.ylabel('Revenue ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot growth rates
        plt.subplot(1, 2, 2)
        plt.bar(range(len(qoq_growth)), qoq_growth.values, alpha=0.8, color='lightgreen', edgecolor='darkgreen')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        plt.xticks(range(len(qoq_growth)), [f"Q{q.quarter}\n{q.year}" for q in qoq_growth.index], rotation=45)
        plt.title('Quarter-over-Quarter Growth', fontweight='bold')
        plt.ylabel('Growth Rate (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig = plt.gcf()
        
        # Create analysis report
        next_quarter = quarterly_revenue.index[-1] + pd.DateOffset(months=3)
        avg_quarterly_growth = qoq_growth.mean()
        
        report = f"""
📊 QUARTERLY TRENDS ANALYSIS
══════════════════════════════

🎯 Query: {query}
📅 Analysis Period: Q{quarterly_revenue.index[0].quarter} {quarterly_revenue.index[0].year} to Q{quarterly_revenue.index[-1].quarter} {quarterly_revenue.index[-1].year}

📈 QUARTERLY PERFORMANCE:
• Total quarters analyzed: {len(quarterly_revenue)}
• Latest quarter revenue: ${quarterly_revenue.iloc[-1]:,.2f}
• Average quarterly revenue: ${quarterly_revenue.mean():,.2f}
• Best quarter: ${quarterly_revenue.max():,.2f}
• Worst quarter: ${quarterly_revenue.min():,.2f}

📊 GROWTH TRENDS:
• Average QoQ growth: {avg_quarterly_growth:+.2f}%
• Growth quarters: {(qoq_growth > 0).sum()}/{len(qoq_growth)}
• Strongest growth: {qoq_growth.max():+.2f}%
• Weakest performance: {qoq_growth.min():+.2f}%

🔮 NEXT QUARTER FORECAST:
• Predicted Q{next_quarter.quarter} {next_quarter.year} revenue: ${next_quarter_forecast:,.2f}
• Expected growth: {((next_quarter_forecast/quarterly_revenue.iloc[-1] - 1) * 100):+.2f}%
• Confidence level: {"High" if len(quarterly_revenue) >= 4 else "Medium" if len(quarterly_revenue) >= 3 else "Low"}

🎯 QUARTERLY INSIGHTS:
• Seasonal pattern: {"Detected" if quarterly_revenue.std() > quarterly_revenue.mean() * 0.2 else "Minimal"}
• Trend direction: {"Upward" if avg_quarterly_growth > 2 else "Stable" if avg_quarterly_growth > -2 else "Downward"}
• Business cycle stage: {"Growth" if avg_quarterly_growth > 5 else "Mature" if avg_quarterly_growth > 0 else "Declining"}

Data Source: {data_source}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return True, report.strip(), fig, None
        
    except Exception as e:
        return False, f"Error in quarterly trends analysis: {str(e)}", None, None

def _predict_six_months(data, query, data_source):
    """Predict 6-month forecast"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    try:
        # Calculate monthly revenue
        monthly_revenue = data.groupby(pd.Grouper(key='order_date', freq='ME'))['amount'].sum()
        monthly_revenue = monthly_revenue[monthly_revenue > 0]
        
        if len(monthly_revenue) < 3:
            return False, "Insufficient data for 6-month forecast", None, None
        
        # Create 6-month forecast using multiple methods
        forecast_months = 6
        forecast_dates = pd.date_range(
            start=monthly_revenue.index[-1] + pd.DateOffset(months=1),
            periods=forecast_months,
            freq='ME'
        )
        
        # Method 1: Linear trend
        x = np.arange(len(monthly_revenue))
        y = monthly_revenue.values
        slope, intercept = np.polyfit(x, y, 1)
        
        trend_forecast = []
        for i in range(forecast_months):
            forecast_value = slope * (len(monthly_revenue) + i) + intercept
            trend_forecast.append(max(0, forecast_value))  # Ensure non-negative
        
        # Method 2: Moving average with growth
        ma_window = min(3, len(monthly_revenue))
        recent_avg = monthly_revenue.tail(ma_window).mean()
        growth_rate = monthly_revenue.pct_change().tail(ma_window).mean()
        
        ma_forecast = []
        current_value = recent_avg
        for i in range(forecast_months):
            current_value = current_value * (1 + growth_rate)
            ma_forecast.append(max(0, current_value))
        
        # Method 3: Seasonal adjustment (if enough data)
        if len(monthly_revenue) >= 12:
            # Simple seasonal pattern
            monthly_ratios = []
            for month in range(1, 13):
                month_data = monthly_revenue[monthly_revenue.index.month == month]
                if len(month_data) > 0:
                    ratio = month_data.mean() / monthly_revenue.mean()
                    monthly_ratios.append(ratio)
                else:
                    monthly_ratios.append(1.0)
            
            seasonal_forecast = []
            base_value = monthly_revenue.tail(3).mean()
            for i, date in enumerate(forecast_dates):
                seasonal_factor = monthly_ratios[date.month - 1]
                seasonal_value = base_value * seasonal_factor * (1 + growth_rate) ** (i + 1)
                seasonal_forecast.append(max(0, seasonal_value))
        else:
            seasonal_forecast = ma_forecast
        
        # Ensemble forecast (average of methods)
        ensemble_forecast = []
        for i in range(forecast_months):
            avg_value = (trend_forecast[i] + ma_forecast[i] + seasonal_forecast[i]) / 3
            ensemble_forecast.append(avg_value)
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Historical and forecast data
        plt.subplot(2, 1, 1)
        
        # Plot historical data
        plt.plot(monthly_revenue.index, monthly_revenue.values, 
                marker='o', linewidth=2, label='Historical Revenue', color='blue')
        
        # Plot forecasts
        plt.plot(forecast_dates, trend_forecast, 
                marker='s', linewidth=2, label='Linear Trend', color='green', alpha=0.8)
        plt.plot(forecast_dates, ma_forecast, 
                marker='^', linewidth=2, label='Moving Average', color='orange', alpha=0.8)
        plt.plot(forecast_dates, ensemble_forecast, 
                marker='*', linewidth=3, label='Ensemble Forecast', color='red', markersize=8)
        
        # Add vertical line to separate historical from forecast
        plt.axvline(x=monthly_revenue.index[-1], color='gray', linestyle='--', alpha=0.5)
        plt.text(monthly_revenue.index[-1], plt.ylim()[1]*0.9, 'Forecast Start', 
                rotation=90, verticalalignment='top', horizontalalignment='right')
        
        plt.title('6-Month Revenue Forecast', fontsize=16, fontweight='bold')
        plt.ylabel('Revenue ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Monthly breakdown
        plt.subplot(2, 1, 2)
        
        months = [date.strftime('%b %Y') for date in forecast_dates]
        x_pos = np.arange(len(months))
        
        width = 0.25
        plt.bar(x_pos - width, trend_forecast, width, label='Linear Trend', alpha=0.8, color='green')
        plt.bar(x_pos, ma_forecast, width, label='Moving Average', alpha=0.8, color='orange')
        plt.bar(x_pos + width, ensemble_forecast, width, label='Ensemble', alpha=0.8, color='red')
        
        plt.title('6-Month Forecast Breakdown', fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Revenue ($)')
        plt.xticks(x_pos, months, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig = plt.gcf()
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'linear_trend': trend_forecast,
            'moving_average': ma_forecast,
            'seasonal_adjusted': seasonal_forecast,
            'ensemble_forecast': ensemble_forecast
        })
        
        # Calculate metrics
        total_forecast = sum(ensemble_forecast)
        avg_monthly_forecast = total_forecast / forecast_months
        current_avg = monthly_revenue.tail(3).mean()
        forecast_growth = ((avg_monthly_forecast / current_avg - 1) * 100)
        
        # Create analysis report
        report = f"""
🔮 6-MONTH REVENUE FORECAST
═══════════════════════════════

🎯 Query: {query}
📅 Forecast Period: {forecast_dates[0].strftime('%B %Y')} to {forecast_dates[-1].strftime('%B %Y')}
📊 Historical Reference: {monthly_revenue.index[0].strftime('%B %Y')} to {monthly_revenue.index[-1].strftime('%B %Y')}

💰 FORECAST SUMMARY:
• Total 6-month revenue: ${total_forecast:,.2f}
• Average monthly revenue: ${avg_monthly_forecast:,.2f}
• Expected growth vs recent avg: {forecast_growth:+.2f}%
• Range: ${min(ensemble_forecast):,.2f} - ${max(ensemble_forecast):,.2f}

📈 MONTHLY BREAKDOWN:
"""
        
        for i, (date, value) in enumerate(zip(forecast_dates, ensemble_forecast)):
            month_change = ((value / ensemble_forecast[i-1] - 1) * 100) if i > 0 else 0
            report += f"• {date.strftime('%B %Y')}: ${value:,.2f}"
            if i > 0:
                report += f" ({month_change:+.1f}%)"
            report += "\n"
        
        report += f"""
🔍 FORECAST CONFIDENCE:
• Data quality: {"High" if len(monthly_revenue) >= 12 else "Medium" if len(monthly_revenue) >= 6 else "Low"}
• Method agreement: {"High" if max(ensemble_forecast)/min(ensemble_forecast) < 1.5 else "Medium"}
• Trend stability: {"Stable" if monthly_revenue.tail(3).std() < monthly_revenue.tail(3).mean() * 0.3 else "Variable"}

📊 KEY INSIGHTS:
• Strongest month: {forecast_dates[ensemble_forecast.index(max(ensemble_forecast))].strftime('%B %Y')} (${max(ensemble_forecast):,.2f})
• Peak season indicator: {"Yes" if max(ensemble_forecast) > avg_monthly_forecast * 1.2 else "No clear pattern"}
• Growth trajectory: {"Accelerating" if ensemble_forecast[-1] > ensemble_forecast[0] * 1.1 else "Steady" if ensemble_forecast[-1] > ensemble_forecast[0] * 0.9 else "Declining"}

🎯 BUSINESS RECOMMENDATIONS:
• Budget planning: Use ${avg_monthly_forecast:,.2f} as monthly baseline
• Cash flow: Prepare for {"seasonal variations" if max(ensemble_forecast)/min(ensemble_forecast) > 1.3 else "stable monthly flow"}
• Strategic focus: {"Revenue acceleration needed" if forecast_growth < 0 else "Maintain current momentum" if forecast_growth < 5 else "Capitalize on growth trend"}

Data Source: {data_source}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return True, report.strip(), fig, forecast_df
        
    except Exception as e:
        return False, f"Error in 6-month forecast: {str(e)}", None, None

def _predict_sales_trends(data, query, data_source):
    """Predict sales trends analysis"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    try:
        # Prepare data for multiple analyses
        data['order_date'] = pd.to_datetime(data['order_date'])
        
        # Monthly sales volume and revenue
        monthly_stats = data.groupby(pd.Grouper(key='order_date', freq='ME')).agg({
            'amount': ['sum', 'count', 'mean'],
            'order_id': 'nunique'
        }).round(2)
        
        monthly_stats.columns = ['total_revenue', 'total_transactions', 'avg_transaction', 'unique_orders']
        monthly_stats = monthly_stats[monthly_stats['total_revenue'] > 0]
        
        if len(monthly_stats) < 3:
            return False, "Insufficient data for sales trends analysis", None, None
        
        # Calculate trends
        revenue_trend = monthly_stats['total_revenue'].pct_change() * 100
        volume_trend = monthly_stats['total_transactions'].pct_change() * 100
        avg_ticket_trend = monthly_stats['avg_transaction'].pct_change() * 100
        
        # Create comprehensive visualization
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Revenue trend
        plt.subplot(3, 2, 1)
        plt.plot(monthly_stats.index, monthly_stats['total_revenue'], marker='o', linewidth=2, color='blue')
        plt.title('Revenue Trend Over Time', fontweight='bold')
        plt.ylabel('Revenue ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Transaction volume
        plt.subplot(3, 2, 2)
        plt.plot(monthly_stats.index, monthly_stats['total_transactions'], marker='s', linewidth=2, color='green')
        plt.title('Transaction Volume Trend', fontweight='bold')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Average transaction value
        plt.subplot(3, 2, 3)
        plt.plot(monthly_stats.index, monthly_stats['avg_transaction'], marker='^', linewidth=2, color='orange')
        plt.title('Average Transaction Value Trend', fontweight='bold')
        plt.ylabel('Average Transaction ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Growth rates
        plt.subplot(3, 2, 4)
        plt.bar(revenue_trend.index[1:], revenue_trend.dropna(), alpha=0.7, label='Revenue Growth', color='blue')
        plt.bar(volume_trend.index[1:], volume_trend.dropna(), alpha=0.7, label='Volume Growth', color='green')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        plt.title('Month-over-Month Growth Rates', fontweight='bold')
        plt.ylabel('Growth Rate (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Correlation analysis
        plt.subplot(3, 2, 5)
        plt.scatter(monthly_stats['total_transactions'], monthly_stats['total_revenue'], 
                   s=100, alpha=0.7, color='purple')
        plt.xlabel('Transaction Volume')
        plt.ylabel('Revenue ($)')
        plt.title('Volume vs Revenue Correlation', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(monthly_stats) > 2:
            z = np.polyfit(monthly_stats['total_transactions'], monthly_stats['total_revenue'], 1)
            p = np.poly1d(z)
            plt.plot(monthly_stats['total_transactions'], p(monthly_stats['total_transactions']), 
                    "r--", alpha=0.8)
        
        # Plot 6: Seasonal patterns (if enough data)
        plt.subplot(3, 2, 6)
        if len(monthly_stats) >= 12:
            monthly_stats['month'] = monthly_stats.index.month
            seasonal_pattern = monthly_stats.groupby('month')['total_revenue'].mean()
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.bar(range(1, 13), [seasonal_pattern.get(i, 0) for i in range(1, 13)], 
                   alpha=0.7, color='coral')
            plt.xticks(range(1, 13), months, rotation=45)
            plt.title('Seasonal Revenue Pattern', fontweight='bold')
            plt.ylabel('Average Revenue ($)')
        else:
            plt.text(0.5, 0.5, 'Insufficient data\nfor seasonal analysis\n(need 12+ months)', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Seasonal Analysis', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig = plt.gcf()
        
        # Calculate key metrics
        total_growth = ((monthly_stats['total_revenue'].iloc[-1] / monthly_stats['total_revenue'].iloc[0] - 1) * 100)
        avg_monthly_growth = revenue_trend.dropna().mean()
        volume_growth = ((monthly_stats['total_transactions'].iloc[-1] / monthly_stats['total_transactions'].iloc[0] - 1) * 100)
        ticket_growth = ((monthly_stats['avg_transaction'].iloc[-1] / monthly_stats['avg_transaction'].iloc[0] - 1) * 100)
        
        # Correlation coefficient
        correlation = monthly_stats['total_transactions'].corr(monthly_stats['total_revenue'])
        
        # Trend direction
        recent_revenue_trend = "increasing" if monthly_stats['total_revenue'].tail(3).mean() > monthly_stats['total_revenue'].head(3).mean() else "decreasing"
        recent_volume_trend = "increasing" if monthly_stats['total_transactions'].tail(3).mean() > monthly_stats['total_transactions'].head(3).mean() else "decreasing"
        
        # Create analysis report
        report = f"""
📈 SALES TRENDS COMPREHENSIVE ANALYSIS
════════════════════════════════════════

🎯 Query: {query}
📅 Analysis Period: {monthly_stats.index[0].strftime('%B %Y')} to {monthly_stats.index[-1].strftime('%B %Y')}

💰 REVENUE TRENDS:
• Total period growth: {total_growth:+.2f}%
• Average monthly growth: {avg_monthly_growth:+.2f}%
• Current monthly revenue: ${monthly_stats['total_revenue'].iloc[-1]:,.2f}
• Revenue trend direction: {recent_revenue_trend.title()}
• Best month: ${monthly_stats['total_revenue'].max():,.2f}
• Worst month: ${monthly_stats['total_revenue'].min():,.2f}

📊 VOLUME TRENDS:
• Transaction volume growth: {volume_growth:+.2f}%
• Current monthly transactions: {monthly_stats['total_transactions'].iloc[-1]:,.0f}
• Volume trend direction: {recent_volume_trend.title()}
• Peak volume month: {monthly_stats['total_transactions'].max():,.0f} transactions
• Average transactions per month: {monthly_stats['total_transactions'].mean():,.0f}

💳 TRANSACTION VALUE TRENDS:
• Average ticket growth: {ticket_growth:+.2f}%
• Current avg transaction: ${monthly_stats['avg_transaction'].iloc[-1]:,.2f}
• Highest avg ticket: ${monthly_stats['avg_transaction'].max():,.2f}
• Lowest avg ticket: ${monthly_stats['avg_transaction'].min():,.2f}

🔍 CORRELATION INSIGHTS:
• Volume-Revenue correlation: {correlation:.3f} ({"Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"})
• Primary growth driver: {"Transaction Volume" if abs(volume_growth) > abs(ticket_growth) else "Transaction Value"}
• Growth consistency: {"High" if revenue_trend.dropna().std() < 10 else "Medium" if revenue_trend.dropna().std() < 20 else "Low"}

📈 TREND PATTERNS:
• Revenue volatility: {monthly_stats['total_revenue'].std()/monthly_stats['total_revenue'].mean()*100:.1f}% coefficient of variation
• Growth momentum: {"Accelerating" if avg_monthly_growth > 0 and revenue_trend.dropna().tail(3).mean() > revenue_trend.dropna().head(3).mean() else "Decelerating" if avg_monthly_growth > 0 else "Declining"}
• Seasonality: {"Detected" if len(monthly_stats) >= 12 and monthly_stats['total_revenue'].std() > monthly_stats['total_revenue'].mean() * 0.3 else "Insufficient data" if len(monthly_stats) < 12 else "Minimal"}

🎯 BUSINESS INSIGHTS:
• Sales performance: {"Excellent" if avg_monthly_growth > 5 else "Good" if avg_monthly_growth > 2 else "Stable" if avg_monthly_growth > -2 else "Needs attention"}
• Focus area: {"Customer acquisition" if volume_growth < ticket_growth else "Customer value optimization" if ticket_growth < volume_growth else "Balanced growth strategy"}
• Market position: {"Expanding" if total_growth > 20 else "Growing" if total_growth > 10 else "Stable" if total_growth > 0 else "Contracting"}

🔮 RECOMMENDATIONS:
• Strategic priority: {"Scale operations for growth" if avg_monthly_growth > 5 else "Optimize current performance" if avg_monthly_growth > 0 else "Implement turnaround strategies"}
• Tactical focus: {"Increase transaction frequency" if volume_growth < 0 else "Improve average order value" if ticket_growth < 0 else "Maintain current momentum"}
• Resource allocation: {"Sales expansion" if volume_growth > ticket_growth else "Product/pricing optimization" if ticket_growth > volume_growth else "Balanced investment"}

Data Source: {data_source}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return True, report.strip(), fig, monthly_stats
        
    except Exception as e:
        return False, f"Error in sales trends analysis: {str(e)}", None, None