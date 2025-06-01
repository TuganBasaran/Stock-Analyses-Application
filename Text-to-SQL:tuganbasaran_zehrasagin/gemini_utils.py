import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

def generate_sql(prompt):
    system_message = (
        "You are a helpful and a SQL assistant. "
        "Use ONLY the provided company documentation to answer the user's question. "
        "If the answer is found — even if it is short — give it clearly. "
        "If the answer is not in the documentation, say: 'The information is not available in the document."
        "Given a natural language question, convert it into an SQL query"
        "for a table called 'orders' with columns: order_id, customer_name, order_date, amount."
        "Do NOT return any explanation—only valid SQL.'"
    )

    response = model.generate_content([system_message, prompt])
    return response.text.strip()

#rag için olan kısımda farklı prompt gerekiyordu ama o zaman sql bozuluyordu ben de ekleme yaptım birleştirdim ikisini
""" SQL--
def generate_sql_only(prompt):
    system_message = (
        "You are an expert SQL assistant. "
        "Given a natural language question, convert it into an SQL query "
        "for a table called 'orders' with columns: order_id, customer_name, order_date, amount. "
        "Do NOT return any explanation—only valid SQL."
    )
    response = model.generate_content([system_message, prompt])
    return response.text.strip()
"""

