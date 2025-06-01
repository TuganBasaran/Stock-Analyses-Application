import streamlit as st
import pandas as pd
import numpy as np
from main_RAG import (
    generate_sql_query, 
    classify_query, 
    answer_company_query,
    load_company_info,
    clear_conversation_history,
    execute_full_llm_query,
    get_next_query_result_filename,
    generate_prediction
)

# Import the array length mismatch fix
try:
    from array_fix import apply_array_fix
    # Apply the fix to handle array length mismatches
    apply_array_fix()
except ImportError:
    print("⚠️ Array length mismatch fix not available")
from db.db import execute_query, get_query_history, clear_query_history, session, add_order
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import sqlalchemy as db
from sqlalchemy.sql import text
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import uuid
import glob
import json
from pathlib import Path
import re

def render_figure(fig, plotly_fig=None):
    """
    Displays matplotlib or Plotly figures in Streamlit interface.
    With enhanced error handling to prevent "Arrays must be same length" errors.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure, plotly.graph_objs._figure.Figure, or None
        Can be either a matplotlib Figure or a Plotly figure object
    plotly_fig : plotly.graph_objs._figure.Figure or None
        Explicit Plotly figure parameter (optional)
    """
    try:
        print(f"🔍 render_figure called with fig={type(fig)}, plotly_fig={type(plotly_fig)}")
        
        # Check for Plotly figure first (from enhanced prediction engine)
        import plotly.graph_objects as go
        
        if fig is not None:
            # Check if it's a Plotly figure
            if hasattr(fig, '_grid_ref') or hasattr(fig, 'data') or 'plotly' in str(type(fig)).lower():
                print("📊 Rendering Plotly figure from enhanced prediction engine")
                st.plotly_chart(fig, use_container_width=True)
                print("✅ Plotly figure rendered successfully")
                return
            
            # Check if it's a matplotlib figure
            import matplotlib.figure
            if isinstance(fig, matplotlib.figure.Figure):
                print(f"📊 Rendering matplotlib figure with {len(fig.get_axes()) if hasattr(fig, 'get_axes') else 0} axes")
                st.pyplot(fig)
                print("✅ Matplotlib figure rendered successfully")
                return
            
            # Try to render as-is (fallback)
            print(f"📊 Attempting to render unknown figure type: {type(fig)}")
            try:
                st.plotly_chart(fig, use_container_width=True)
                print("✅ Figure rendered as Plotly")
                return
            except:
                st.pyplot(fig)
                print("✅ Figure rendered as matplotlib")
                return
                
        elif plotly_fig is not None:
            print("📊 Rendering explicit Plotly figure")
            st.plotly_chart(plotly_fig, use_container_width=True)
            print("✅ Plotly figure rendered successfully")
            return
        else:
            print("⚠️ No figure provided to render_figure")
            st.write("🔍 No visualization available.")
            
    except Exception as e:
        import traceback
        print(f"❌ Error rendering figure: {str(e)}")
        traceback.print_exc()
        st.error(f"Failed to display visualization: {str(e)}")
        
        # Show a fallback text instead
        st.write("⚠️ The visualization could not be displayed due to a technical issue.")
        st.write("Please try a different query or visualization type.")

# Enhanced Chat Functions for Human-like Interaction

def analyze_sentiment(text):
    """Analyze sentiment of user input using LLM prompt"""
    try:
        from main_RAG import model
        
        prompt = f"""
        Analyze the sentiment of the following user message. Provide a detailed analysis in JSON format.
        
        User Message: "{text}"
        
        Please analyze:
        1. Overall sentiment (positive, negative, neutral)
        2. Emotional tone (excited, frustrated, curious, urgent, casual, formal, etc.)
        3. Confidence level (0.0 to 1.0)
        4. Intent indicators (question, request, complaint, compliment, etc.)
        5. Politeness level (polite, neutral, rude)
        
        Return ONLY a valid JSON object with this structure:
        {{
            "sentiment": "positive|negative|neutral",
            "emotional_tone": "description of emotional tone",
            "confidence": 0.0-1.0,
            "intent_type": "question|request|complaint|compliment|other",
            "politeness": "polite|neutral|rude",
            "urgency": "high|medium|low",
            "explanation": "brief explanation of the analysis"
        }}
        """
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 300,
                "top_p": 0.95,
            }
        )
        
        # Parse JSON response
        import json
        sentiment_data = json.loads(response.text.strip())
        
        # Validate required fields
        required_fields = ["sentiment", "emotional_tone", "confidence", "intent_type", "politeness", "urgency"]
        for field in required_fields:
            if field not in sentiment_data:
                sentiment_data[field] = "neutral" if field in ["sentiment", "politeness"] else "unknown"
        
        # Ensure confidence is a float
        if not isinstance(sentiment_data["confidence"], (int, float)):
            sentiment_data["confidence"] = 0.5
        
        return sentiment_data
        
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        # Fallback to simple analysis
        return {
            "sentiment": "neutral",
            "emotional_tone": "casual",
            "confidence": 0.5,
            "intent_type": "question",
            "politeness": "neutral",
            "urgency": "medium",
            "explanation": "Fallback analysis due to error"
        }

def get_conversation_context(chat_history, max_context=5):
    """Extract relevant context from recent conversation history"""
    if not chat_history or len(chat_history) < 2:
        return ""
    
    # Get last few exchanges
    recent_messages = chat_history[-max_context:]
    context_parts = []
    
    for msg in recent_messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'user':
            context_parts.append(f"User asked: {content}")
        else:
            # Summarize assistant responses
            if len(content) > 100:
                content = content[:100] + "..."
            context_parts.append(f"Assistant responded: {content}")
            
            # Include data context if available
            if 'dataframe' in msg and not msg['dataframe'].empty:
                df = msg['dataframe']
                context_parts.append(f"Data context: {len(df)} rows with columns {list(df.columns)}")
    
    return "\n".join(context_parts)

def generate_human_like_response(user_query, sentiment_info, context, query_result=None):
    """Generate a more human-like response based on advanced sentiment analysis and context"""
    try:
        from main_RAG import model
        
        # Create a context-aware prompt for response generation
        prompt = f"""
        Generate a natural, human-like response greeting based on the user's query and sentiment analysis.
        
        User Query: "{user_query}"
        
        Sentiment Analysis:
        - Overall Sentiment: {sentiment_info.get('sentiment', 'neutral')}
        - Emotional Tone: {sentiment_info.get('emotional_tone', 'casual')}
        - Intent Type: {sentiment_info.get('intent_type', 'question')}
        - Politeness Level: {sentiment_info.get('politeness', 'neutral')}
        - Urgency: {sentiment_info.get('urgency', 'medium')}
        - Confidence: {sentiment_info.get('confidence', 0.5)}
        
        Conversation Context: {context if context else "No previous context"}
        
        Query Result Info: {f"Found {len(query_result)} records" if isinstance(query_result, pd.DataFrame) and not query_result.empty else "No specific result data"}
        
        Generate a natural, friendly response that:
        1. Matches the user's emotional tone and politeness level
        2. Acknowledges their sentiment appropriately
        3. References previous context if relevant
        4. Is conversational and warm, not robotic
        5. Adapts to the urgency level
        6. Uses appropriate language for the intent type
        
        Keep the response to 1-2 sentences maximum. Make it sound like a helpful human assistant.
        Return only the response text, no additional formatting.
        """
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 150,
                "top_p": 0.9,
            }
        )
        
        return response.text.strip()
        
    except Exception as e:
        print(f"Error generating human-like response: {str(e)}")
        
        # Fallback to rule-based response
        sentiment = sentiment_info.get("sentiment", "neutral")
        emotional_tone = sentiment_info.get("emotional_tone", "casual")
        urgency = sentiment_info.get("urgency", "medium")
        
        if sentiment == "positive":
            if urgency == "high":
                greeting_phrases = ["Absolutely! Let me help you right away!", "Great question! I'll get that for you immediately!"]
            else:
                greeting_phrases = ["Great question!", "I'd be happy to help!", "Excellent! Let me help you with that."]
        elif sentiment == "negative":
            if "frustrated" in emotional_tone:
                greeting_phrases = ["I understand your frustration. Let me help resolve this.", "I hear your concern and I'm here to help."]
            else:
                greeting_phrases = ["I understand your concern.", "Let me help clarify that for you.", "No worries, let me assist you."]
        else:
            if urgency == "high":
                greeting_phrases = ["I'll get that information for you right away.", "Let me quickly analyze that for you."]
            else:
                greeting_phrases = ["Let me help you with that.", "I'll look into that for you.", "Let me analyze that."]
        
        import random
        return random.choice(greeting_phrases)

def detect_query_intent(user_query, chat_history):
    """Detect the intent behind the user's query using LLM analysis"""
    try:
        from main_RAG import model
        
        # Get conversation context for better intent detection
        context = get_conversation_context(chat_history, max_context=3) if chat_history else "No previous context"
        
        prompt = f"""
        Analyze the user's query intent based on the query text and conversation context.
        
        User Query: "{user_query}"
        Conversation Context: {context}
        
        Determine the following intents and return a JSON object:
        
        1. is_follow_up: Is this a follow-up question to previous conversation? (true/false)
        2. wants_visualization: Does the user want a chart, graph, or visualization? (true/false)
        3. wants_comparison: Does the user want to compare different data points? (true/false)
        4. wants_trend: Does the user want trend analysis over time? (true/false)
        5. wants_explanation: Does the user want explanation of previous results? (true/false)
        6. wants_prediction: Does the user want future predictions or forecasts? (true/false)
        7. query_type: What type of query is this? (data_query, company_info, visualization_request, explanation_request, other)
        8. urgency_level: How urgent does this query seem? (low, medium, high)
        9. has_context: Does this query reference previous conversation? (true/false)
        
        Return ONLY a valid JSON object with these fields.
        """
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 200,
                "top_p": 0.95,
            }
        )
        
        # Parse JSON response
        import json
        intent_data = json.loads(response.text.strip())
        
        # Validate and set defaults for missing fields
        default_intent = {
            "is_follow_up": False,
            "wants_visualization": False,
            "wants_comparison": False,
            "wants_trend": False,
            "wants_explanation": False,
            "wants_prediction": False,
            "query_type": "other",
            "urgency_level": "medium",
            "has_context": len(chat_history) > 2
        }
        
        # Merge with defaults
        for key, default_value in default_intent.items():
            if key not in intent_data:
                intent_data[key] = default_value
        
        return intent_data
        
    except Exception as e:
        print(f"Error in intent detection: {str(e)}")
        
        # Fallback to rule-based intent detection
        query_lower = user_query.lower()
        
        # Check for follow-up questions
        follow_up_indicators = [
            "what about", "how about", "can you show", "also show", "additionally",
            "more details", "explain more", "why", "how", "what if", "compare"
        ]
        
        is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)
        
        # Check for visualization requests
        viz_requests = [
            "show", "display", "visualize", "chart", "graph", "plot", "diagram"
        ]
        wants_visualization = any(req in query_lower for req in viz_requests)
        
        # Check for comparison requests
        comparison_words = ["compare", "vs", "versus", "difference", "against", "between"]
        wants_comparison = any(word in query_lower for word in comparison_words)
        
        # Check for trend analysis
        trend_words = ["trend", "over time", "monthly", "yearly", "growth", "increase", "decrease"]
        wants_trend = any(word in query_lower for word in trend_words)
        
        # Check for predictions
        prediction_words = ["predict", "forecast", "future", "next", "upcoming", "will be"]
        wants_prediction = any(word in query_lower for word in prediction_words)
        
        # Check for explanations
        explanation_words = ["explain", "why", "how does", "what does", "meaning"]
        wants_explanation = any(word in query_lower for word in explanation_words)
        
        return {
            "is_follow_up": is_follow_up,
            "wants_visualization": wants_visualization,
            "wants_comparison": wants_comparison,
            "wants_trend": wants_trend,
            "wants_explanation": wants_explanation,
            "wants_prediction": wants_prediction,
            "query_type": "data_query" if any(word in query_lower for word in ["show", "get", "find", "select"]) else "other",
            "urgency_level": "high" if any(word in query_lower for word in ["urgent", "quickly", "asap", "immediately"]) else "medium",
            "has_context": len(chat_history) > 2
        }

def enhance_query_with_context(user_query, chat_history):
    """Enhance user query with context from previous conversation using LLM"""
    if not chat_history or len(chat_history) < 2:
        return user_query
    
    try:
        from main_RAG import model
        
        # Get the last few messages for context
        recent_context = get_conversation_context(chat_history, max_context=3)
        
        # Check if this is a follow-up question
        intent = detect_query_intent(user_query, chat_history)
        
        if intent.get("is_follow_up", False) or intent.get("has_context", False):
            
            # Find the last data result for additional context
            last_data_context = ""
            for msg in reversed(chat_history):
                if 'dataframe' in msg and not msg['dataframe'].empty:
                    df = msg['dataframe']
                    columns = ", ".join(df.columns)
                    sample_data = df.head(2).to_string() if len(df) > 0 else "No data"
                    last_data_context = f"Previous data context: Table with columns: {columns}\nSample data:\n{sample_data}"
                    break
            
            prompt = f"""
            The user has asked a follow-up question that references previous conversation context. 
            Enhance their query to include relevant context for better understanding.
            
            Original User Query: "{user_query}"
            
            Previous Conversation Context:
            {recent_context}
            
            {last_data_context}
            
            Intent Analysis:
            - Is Follow-up: {intent.get('is_follow_up', False)}
            - Wants Visualization: {intent.get('wants_visualization', False)}
            - Wants Comparison: {intent.get('wants_comparison', False)}
            - Wants Explanation: {intent.get('wants_explanation', False)}
            
            Please enhance the user's query by:
            1. Adding relevant context from previous conversation
            2. Making implicit references explicit
            3. Clarifying what data or visualization they're referring to
            4. Maintaining the user's original intent
            
            Return an enhanced query that includes the necessary context for better processing.
            Keep it natural and conversational.
            """
            
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 300,
                    "top_p": 0.9,
                }
            )
            
            enhanced_query = response.text.strip()
            
            # Make sure the enhanced query is reasonable and not too long
            if len(enhanced_query) > 500:
                enhanced_query = enhanced_query[:500] + "..."
            
            return enhanced_query
            
    except Exception as e:
        print(f"Error enhancing query with context: {str(e)}")
    
    # Fallback to simple enhancement
    recent_context = get_conversation_context(chat_history, max_context=3)
    intent = detect_query_intent(user_query, chat_history)
    
    if intent.get("is_follow_up", False):
        # Find the last data result
        last_data_context = ""
        for msg in reversed(chat_history):
            if 'dataframe' in msg and not msg['dataframe'].empty:
                df = msg['dataframe']
                columns = ", ".join(df.columns)
                last_data_context = f"\nPrevious data context: Table with columns: {columns}"
                break
        
        enhanced_query = f"""
        User Query: {user_query}
        
        Previous Conversation Context:
        {recent_context}
        {last_data_context}
        
        This appears to be a follow-up question. Please consider the previous context when generating the response.
        """
        
        return enhanced_query
    
    return user_query

def enhance_prediction_query(user_query, chat_history, data_context=None):
    """Enhance prediction queries with advanced LLM analysis for better forecasting context"""
    try:
        from main_RAG import model
        
        # Get conversation context
        recent_context = get_conversation_context(chat_history, max_context=3)
        
        # Analyze data context if available
        data_info = ""
        if data_context and hasattr(data_context, 'columns'):
            columns = ", ".join(data_context.columns)
            data_summary = f"Available data columns: {columns}"
            if len(data_context) > 0:
                date_cols = [col for col in data_context.columns if 'date' in col.lower()]
                amount_cols = [col for col in data_context.columns if any(term in col.lower() for term in ['amount', 'price', 'revenue', 'cost', 'value'])]
                data_info = f"""
                Data Summary:
                - Total records: {len(data_context)}
                - Date columns: {date_cols}
                - Value columns: {amount_cols}
                - {data_summary}
                """
        
        prompt = f"""
        You are an expert in business forecasting and predictive analytics. 
        Analyze this prediction query and enhance it with sophisticated forecasting context.
        
        Original Query: "{user_query}"
        
        Conversation Context:
        {recent_context}
        
        {data_info}
        
        Please enhance this prediction query by:
        1. Identifying the specific business metric to forecast (revenue, sales volume, customer growth, etc.)
        2. Determining the appropriate time horizon (daily, weekly, monthly, quarterly, yearly)
        3. Suggesting relevant forecasting methodologies based on the data type
        4. Adding business context and seasonal considerations
        5. Specifying confidence intervals and prediction scenarios
        6. Including trend analysis and pattern recognition requirements
        7. Making the query more specific and actionable for time series analysis
        
        Return an enhanced, professional prediction query that will generate better forecasting results.
        Keep it concise but comprehensive, focusing on business value and analytical depth.
        """
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 400,
                "top_p": 0.9,
            }
        )
        
        enhanced_query = response.text.strip()
        
        # Ensure the enhanced query is reasonable
        if len(enhanced_query) > 600:
            enhanced_query = enhanced_query[:600] + "..."
        
        return enhanced_query
        
    except Exception as e:
        print(f"Error enhancing prediction query: {str(e)}")
        return user_query

def generate_prediction_insights(user_query, prediction_output, forecasts_df):
    """Generate intelligent insights and explanations for prediction results using LLM"""
    try:
        from main_RAG import model
        
        # Prepare forecast data summary
        forecast_summary = ""
        if forecasts_df is not None and not forecasts_df.empty:
            forecast_summary = f"""
            Forecast Results Summary:
            - Time period: {forecasts_df.index.min()} to {forecasts_df.index.max()}
            - Number of predictions: {len(forecasts_df)}
            - Forecast columns: {', '.join(forecasts_df.columns)}
            """
            
            # Add statistical summary
            try:
                numeric_cols = forecasts_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats = forecasts_df[numeric_cols].describe()
                    forecast_summary += f"\nStatistical Summary:\n{stats.to_string()}"
            except:
                pass
        
        prompt = f"""
        You are a senior business analyst specializing in predictive analytics and forecasting.
        Analyze the prediction results and generate professional insights and recommendations.
        
        Original User Query: "{user_query}"
        
        Prediction Analysis Output:
        {prediction_output}
        
        {forecast_summary}
        
        Please provide:
        1. **Executive Summary**: Key findings and predictions in business terms
        2. **Methodology Assessment**: Brief explanation of the forecasting approach used
        3. **Key Insights**: What the data reveals about future trends and patterns
        4. **Business Implications**: How these predictions should inform decision-making
        5. **Confidence Assessment**: Reliability and limitations of the predictions
        6. **Actionable Recommendations**: Specific steps based on the forecast results
        
        Format your response in clear, professional English with bullet points where appropriate.
        Focus on business value and actionable insights rather than technical details.
        Keep the response concise but comprehensive (max 300 words).
        """
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 500,
                "top_p": 0.9,
            }
        )
        
        return response.text.strip()
        
    except Exception as e:
        print(f"Error generating prediction insights: {str(e)}")
        return "Prediction analysis completed successfully. Please review the forecast results above for detailed insights."

def generate_smart_suggestions(chat_history):
    """Generate smart follow-up suggestions based on conversation history using LLM"""
    if not chat_history or len(chat_history) < 2:
        return []
    
    try:
        from main_RAG import model
        
        # Get recent conversation context
        recent_context = get_conversation_context(chat_history, max_context=5)
        
        # Get information about last data result
        last_data_info = "No previous data available"
        for msg in reversed(chat_history):
            if 'dataframe' in msg and not msg['dataframe'].empty:
                df = msg['dataframe']
                columns = list(df.columns)
                row_count = len(df)
                
                # Get sample data for better context
                sample_data = df.head(3).to_dict('records') if len(df) > 0 else []
                
                last_data_info = f"""
                Last query returned {row_count} rows with columns: {columns}
                Sample data: {sample_data}
                """
                break
        
        prompt = f"""
        Based on the conversation history and available data, generate 4 smart follow-up suggestions 
        that would be most relevant and helpful to the user.
        
        Recent Conversation Context:
        {recent_context}
        
        Available Data Information:
        {last_data_info}
        
        Generate suggestions that:
        1. Build on the current conversation naturally
        2. Offer different types of analysis (visualization, aggregation, comparison, trends)
        3. Are specific to the available data columns
        4. Use clear, actionable language
        5. Are diverse in their approach (don't repeat similar suggestions)
        
        Focus on:
        - Data visualization requests if not already shown
        - Aggregate analysis (averages, totals, counts)
        - Time-based trends if date columns are available
        - Customer or category analysis if relevant columns exist
        - Comparisons between different data segments
        - Company information if no data queries were made recently
        
        Return exactly 4 suggestions as a JSON array of strings.
        Example format: ["Show this as a bar chart", "What's the average revenue?", "Compare monthly trends", "Tell me about the company"]
        """
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.6,
                "max_output_tokens": 300,
                "top_p": 0.9,
            }
        )
        
        # Parse JSON response
        import json
        suggestions = json.loads(response.text.strip())
        
        # Validate that we got a list of strings
        if isinstance(suggestions, list) and len(suggestions) >= 4:
            return suggestions[:4]
        else:
            raise ValueError("Invalid suggestions format")
            
    except Exception as e:
        print(f"Error generating smart suggestions: {str(e)}")
        
        # Fallback to rule-based suggestions
        suggestions = []
        
        # Look for the last successful query result
        for msg in reversed(chat_history):
            if 'dataframe' in msg and not msg['dataframe'].empty:
                df = msg['dataframe']
                
                # Suggest visualizations if not already shown
                if 'fig' not in msg:
                    suggestions.extend([
                        "Can you show this as a chart?",
                        "Create a visualization for this data"
                    ])
                
                # Suggest aggregations based on columns
                if 'customer_name' in df.columns:
                    suggestions.append("Which customer had the highest orders?")
                if 'order_date' in df.columns:
                    suggestions.append("Show me monthly trends")
                if 'price' in df.columns or 'total' in df.columns:
                    suggestions.append("What's the average value?")
                
                break
        
        # Add general suggestions if we don't have enough
        general_suggestions = [
            "Tell me about the company",
            "What's our best selling product?",
            "Show me sales performance",
            "Predict next month's revenue in 2021"
        ]
        
        # Combine and limit to 4
        all_suggestions = suggestions + general_suggestions
        return all_suggestions[:4]

# Page configuration with dark theme
st.set_page_config(
    page_title="Text-to-SQL Chat Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Text-to-SQL Converter with Chat Interface"
    }
)

# Apply dark theme CSS
st.markdown("""
<style>
    /* Override to dark theme */
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stTextInput > div > div > input {
        color: white;
        background-color: #262730;
    }
    /* Chat styling */
    .user-message {
        background-color: #1E88E5;
        color: white;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
        margin-left: auto;
        margin-right: 5px;
    }
    .assistant-message {
        background-color: #262730;
        color: white;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
        margin-left: 5px;
    }
    /* Chart gallery styling */
    .chart-gallery {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 10px;
        margin-bottom: 20px;
    }
    .chart-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #555;
    }
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 5px 15px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    /* Smart suggestion buttons */
    div[data-testid="column"] .stButton>button {
        background-color: #1E88E5;
        color: white;
        border: 1px solid #1976D2;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 0.9em;
        margin: 2px 0;
        transition: all 0.3s ease;
    }
    div[data-testid="column"] .stButton>button:hover {
        background-color: #1976D2;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(30, 136, 229, 0.3);
    }
    /* Prediction example buttons */
    .prediction-buttons .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 20px;
        font-size: 0.95em;
        font-weight: 600;
        margin: 5px 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
    }
    .prediction-buttons .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(78, 205, 196, 0.5);
        background: linear-gradient(45deg, #FF8E85, #6FEDE3);
    }
    /* Adjust visualization container */
    .visualization-container {
        margin-top: 20px;
        margin-bottom: 20px;
    }
    /* Chat input styling */
    div[data-testid="stChatInput"] textarea {
        color: white;
        background-color: #262730;
        border: 1px solid #555;
        border-radius: 20px;
    }
    /* Header styling */
    h1, h2, h3 {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Function to create a direct visualization for chat
def create_visualization(df, user_query, viz_type="bar"):
    """Create visualization directly in Streamlit"""
    if df.empty or len(df) == 0:
        return None
    
    # Find numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) == 0:
        return None
    
    # Choose column_y from numeric columns
    column_y = numeric_columns[0]
    
    # Choose appropriate column for X-axis
    if 'order_date' in df.columns:
        column_x = 'order_date'
    elif 'order_year' in df.columns and 'order_month' in df.columns:
        # Create a date column from year and month
        df['date'] = df.apply(lambda x: f"{int(x['order_year'])}-{int(x['order_month']):02d}", axis=1)
        column_x = 'date'
    elif 'customer_name' in df.columns:
        column_x = 'customer_name'
    else:
        column_x = df.columns[0]
        
    # Create visualization based on selected type
    if viz_type == "bar":
        fig = px.bar(df, x=column_x, y=column_y, title=f"Results for: {user_query}")
        # Style to match dark theme
        fig.update_layout(
            plot_bgcolor='rgba(17, 17, 17, 0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)'),
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            colorway=['#1E88E5']  # Use blue color for bars
        )
        return fig
    elif viz_type == "line":
        fig = px.line(df, x=column_x, y=column_y, title=f"Results for: {user_query}")
        fig.update_layout(
            plot_bgcolor='rgba(17, 17, 17, 0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)'),
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig
    elif viz_type == "pie":
        fig = px.pie(df, names=column_x, values=column_y, title=f"Results for: {user_query}")
        fig.update_layout(
            plot_bgcolor='rgba(17, 17, 17, 0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig
    elif viz_type == "heatmap":
        # Create a correlation matrix if there are multiple numeric columns
        if len(numeric_columns) > 1:
            corr_df = df[numeric_columns].corr()
            fig = px.imshow(corr_df, title="Correlation Matrix")
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            return fig
    return None

# Function to generate explanation for visualizations
def generate_viz_explanation(user_query, data_summary, viz_type):
    """Generates a natural language explanation of a visualization based on the query and data summary"""
    import pandas as pd
    from datetime import datetime
    
    # Remove sensitive information from query
    safe_query = re.sub(r'(password|token|api[_\s]?key)\s*[:=]\s*\S+', r'\1: [REDACTED]', user_query, flags=re.IGNORECASE)
    
    # Generate English explanation by default (Turkish language detection removed)
    
    # Construct prompt for explanation in English only
    prompt = f"""
    Generate a visualization explanation:
    
    User Query: {safe_query}
    Visualization Type: {viz_type}
    Data Summary: {data_summary}
    
    The explanation should:
    1. Be concise and clear (3-5 sentences)
    2. Interpret what the visualization shows
    3. Highlight important trends or insights
    4. Use clear, non-technical language
    5. Be directly relevant to the user's query
    6. Not mention technical details about how the visualization was created
    
    Return only the explanation text in English, no additional formatting.
    """
    
    try:
        # Import only when needed to avoid circular imports
        from main_RAG import model
        
        # Generate the explanation
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 512,
                "top_p": 0.95,
                "top_k": 40,
            }
        )
        explanation = response.text.strip()
        
        # If explanation is too technical or generic, refine it
        if len(explanation) < 50 or "This chart" in explanation:
            # Try again with a more direct prompt
            refine_prompt = f"""
            Write a more detailed explanation for the visualization analyzing the following data:
            
            Query: {safe_query}
            Chart Type: {viz_type}
            Data Summary: {data_summary}
            
            - Tell the main story of the data
            - Point out visible trends or patterns
            - Connect your explanation to the user's query
            - Avoid technical jargon
            """
            
            refine_response = model.generate_content(
                refine_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 512,
                }
            )
            explanation = refine_response.text.strip()
        
        return explanation
        
    except Exception as e:
        print(f"Error generating visualization explanation: {str(e)}")
        # Fallback explanation in English
        return "This visualization displays the results of your query. The data is presented in a graphical format to help identify patterns and insights more easily."

# Function to enhance SQL queries for specific cases
def enhance_sql_query(query, sql_query):
    # For monthly revenue trends
    if "monthly" in query.lower() and ("revenue" in query.lower() or "trend" in query.lower()):
        sql_query = """
        SELECT 
            YEAR(order_date) AS order_year,
            MONTH(order_date) AS order_month,
            SUM(price) AS monthly_revenue
        FROM 
            Orders
        GROUP BY 
            YEAR(order_date), MONTH(order_date)
        ORDER BY 
            order_year, order_month
        """
    
    # For correlation heatmap requests, we need to fetch all data and create the heatmap in Python
    elif "correlation" in query.lower() and "heat map" in query.lower():
        # Instead of trying to calculate correlation in SQL, just fetch all relevant columns
        sql_query = """
        SELECT * FROM Orders
        """
    
    return sql_query

# Initialize session state if needed
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'company_info' not in st.session_state:
    st.session_state.company_info = load_company_info()
    
if 'show_gallery' not in st.session_state:
    st.session_state.show_gallery = False

# Enhanced chat metadata
if 'conversation_metadata' not in st.session_state:
    st.session_state.conversation_metadata = {
        'start_time': datetime.now(),
        'total_queries': 0,
        'sentiment_history': [],
        'last_context': None,
        'user_preferences': {
            'preferred_viz_type': 'bar',
            'response_style': 'friendly'
        }
    }
    
# Main title with enhanced prediction capabilities
st.title("🤖 AI-Powered Text-to-SQL & Prediction Assistant")
st.markdown("### 💡 Ask questions about your data, get insights, and forecast the future!")
st.markdown("**Features:** SQL Generation | Data Analysis | Predictive Forecasting | Company Q&A")

# Sidebar for settings and options
with st.sidebar:
    st.header("Settings")
    
    viz_type = st.selectbox(
        "Default Visualization Type",
        ["bar", "line", "pie", "heatmap"],
        index=0
    )
    
    # Option to show/hide visualization gallery
    if st.button("Show Visualization Gallery"):
        st.session_state.show_gallery = not st.session_state.show_gallery
        st.rerun()

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        clear_conversation_history()
        st.success("Chat history cleared!")
    
    # Help section
    st.subheader("Example Questions")
    st.markdown("""
    **Data Analysis:**
    - "How many orders were placed in January?"
    - "What is the total revenue?"
    - "Which customer spent the most?"
    - "Show monthly revenue trend"
    
    **Predictions & Forecasting:**
    - "Predict next month's revenue"
    - "Forecast sales for next quarter"
    - "What will be the expected growth rate?"
    - "Project revenue for next 6 months"
    
    **Company Information:**
    - "What is the company's mission?"
    - "Tell me about the company"
    """)
    
    # Smart suggestions based on conversation
    if len(st.session_state.chat_history) > 0:
        st.subheader("💡 Smart Suggestions")
        suggestions = generate_smart_suggestions(st.session_state.chat_history)
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{suggestion[:20]}", use_container_width=True):
                # Add the suggestion as user input
                st.session_state.user_input = suggestion
                st.rerun()
    


def extract_visualization_type(query: str) -> str:
    """Extracts the visualization type from the user query string."""
    query = query.lower()
    if "bar chart" in query or "bar graph" in query:
        return "bar"
    elif "line chart" in query or "line graph" in query:
        return "line"
    elif "pie chart" in query or "pie graph" in query or "pie" in query:
        return "pie"
    elif "heatmap" in query or "heat map" in query or "correlation" in query:
        return "heatmap"
    return ""

# Only show the visualization gallery when requested
if st.session_state.show_gallery:
    # Create a section for saved visualizations
    st.header("Saved Visualizations")

    # Visualization gallery is disabled since we no longer save PNG files
    st.info("Visualization files are no longer saved to disk. All visualizations are displayed directly in the chat.")
    
    # Button to hide gallery
    if st.button("Hide Gallery"):
        st.session_state.show_gallery = False
        st.rerun()  # st.experimental_rerun() yerine st.rerun() kullanıyoruz

# Display welcome message for new users
if len(st.session_state.chat_history) == 0:
    st.markdown("""
    ### 👋 Welcome to your AI-Powered Data Assistant!
    
    **What can I help you with today?**
    
    🔍 **Data Analysis**: "How many orders were placed in January?"  
    📈 **Predictions**: "Predict next month's revenue" or "Forecast sales trends"  
    🏢 **Company Info**: "Tell me about the company"  
    📊 **Visualizations**: "Show me monthly revenue as a bar chart"
    
    **Try the prediction buttons below or type your question!**
    """)

# Display chat history
st.header("💬 Chat History")
chat_area = st.container()

with chat_area:
    for message in st.session_state.chat_history:
        role = message.get('role', 'user')  # Default to user if not specified
        
        if role == 'user':
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
            
            # Display SQL if present
            if 'sql' in message:
                with st.chat_message("assistant"):
                    st.code(message['sql'], language="sql")
            
            # Display dataframe if present
            if 'dataframe' in message and not message['dataframe'].empty:
                with st.chat_message("assistant"):
                    st.dataframe(message['dataframe'])
            
            # Display image if present
            if 'image' in message and message['image']:
                with st.chat_message("assistant"):
                    st.image(message['image']) 
            
            # Display figure if present using unified render_figure function
            if 'fig' in message and message['fig'] is not None:
                with st.chat_message("assistant"):
                    # Check if it's a matplotlib figure or Plotly figure
                    import matplotlib.figure
                    if isinstance(message['fig'], matplotlib.figure.Figure):
                        render_figure(message['fig'])  # matplotlib figure
                    else:
                        render_figure(None, message['fig'])  # Plotly figure

# Enhanced User input with prediction examples
user_query = st.chat_input("💬 Ask about your data, company info, or try predictions like 'predict next month revenue' or 'forecast sales trends'...")

# Add quick prediction examples
st.markdown("### 🔮 Quick Prediction Examples")
st.markdown('<div class="prediction-buttons">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📈 Predict Revenue", help="Forecast next month's revenue"):
        st.session_state.user_input = "predict next month's revenue"
        st.rerun()

with col2:
    if st.button("📊 Sales Trends", help="Forecast sales performance"):
        st.session_state.user_input = "forecast sales trends for next quarter"
        st.rerun()

with col3:
    if st.button("🎯 Growth Rate", help="Predict growth patterns"):
        st.session_state.user_input = "what will be the expected growth rate"
        st.rerun()

with col4:
    if st.button("📅 Monthly Forecast", help="6-month revenue projection"):
        st.session_state.user_input = "project revenue for next 6 months"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Handle suggestion clicks
if 'user_input' in st.session_state and st.session_state.user_input:
    user_query = st.session_state.user_input
    del st.session_state.user_input

if user_query:
    # Enhanced chat processing with sentiment analysis and context
    sentiment_info = analyze_sentiment(user_query)
    query_intent = detect_query_intent(user_query, st.session_state.chat_history)
    enhanced_query = enhance_query_with_context(user_query, st.session_state.chat_history)
    
    # Update conversation metadata
    st.session_state.conversation_metadata['total_queries'] += 1
    st.session_state.conversation_metadata['sentiment_history'].append(sentiment_info)
    
    # Generate human-like greeting based on sentiment and context
    human_greeting = generate_human_like_response(user_query, sentiment_info, get_conversation_context(st.session_state.chat_history))
    
    # Check if this is a request to change visualization type or explain a graph
    is_viz_change_request = False
    is_explanation_request = False
    
    # Check if the user is asking for a specific visualization type
    requested_viz_type = extract_visualization_type(user_query)
    if requested_viz_type:
        is_viz_change_request = True
        viz_type = requested_viz_type
        
    # Check if the user is asking for an explanation
    if "explain" in user_query.lower() and any(term in user_query.lower() for term in ["graph", "chart", "visualization", "figure", "plot", "diagram"]):
        is_explanation_request = True
    
    # Add user message to chat history with metadata
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_query,
        'timestamp': datetime.now(),
        'sentiment': sentiment_info,
        'intent': query_intent
    })

    # Display user message immediately
    st.markdown(f"<div class='user-message'>{user_query}</div>", unsafe_allow_html=True)
    
    # Process the query
    with st.spinner("Processing your question..."):
        try:
            # Handle visualization type change requests directly
            if is_viz_change_request and len(st.session_state.chat_history) > 2:
                # Check if we have previous results to visualize in a different way
                for i in range(len(st.session_state.chat_history) - 2, 0, -1):
                    if 'dataframe' in st.session_state.chat_history[i] and not st.session_state.chat_history[i]['dataframe'].empty:
                        # Found previous dataframe, create new visualization
                        df = st.session_state.chat_history[i]['dataframe']
                        try:
                            fig = create_visualization(df, user_query, viz_type)
                            
                            if fig is not None:
                                response_content = f"I've visualized your data as a '{viz_type}' chart:"
                                st.session_state.chat_history.append({
                                    'role': 'assistant',
                                    'content': response_content,
                                    'fig': fig
                                })
                                # Display response immediately
                                st.markdown(f"<div class='assistant-message'>{response_content}</div>", unsafe_allow_html=True)
                                render_figure(None, fig)  # Use unified render_figure for Plotly charts
                            else:
                                response_content = f"Sorry, I cannot visualize your data as a '{viz_type}' chart. Please try another chart type."
                                st.session_state.chat_history.append({
                                    'role': 'assistant',
                                    'content': response_content
                                })
                                # Display response immediately
                                st.markdown(f"<div class='assistant-message'>{response_content}</div>", unsafe_allow_html=True)
                        except Exception as viz_error:
                            # Handle visualization creation errors
                            error_message = f"An error occurred while creating the visualization: {str(viz_error)}"
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': error_message
                            })
                            # Display error immediately
                            st.markdown(f"<div class='assistant-message'>{error_message}</div>", unsafe_allow_html=True)
                            st.error(error_message)
                        break
                else:
                    # No previous dataframe found
                    response_content = "No previous data found to visualize. Please run a query that returns data first."
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response_content
                    })
                    # Display response immediately
                    st.markdown(f"<div class='assistant-message'>{response_content}</div>", unsafe_allow_html=True)
            
            # Handle explanation requests
            elif is_explanation_request and len(st.session_state.chat_history) > 2:
                # Look for the last visualization to explain
                for i in range(len(st.session_state.chat_history) - 2, 0, -1):
                    if ('fig' in st.session_state.chat_history[i] and st.session_state.chat_history[i]['fig'] is not None) or \
                       ('image' in st.session_state.chat_history[i] and st.session_state.chat_history[i]['image'] is not None):
                        # Found visualization to explain
                        
                        # Get relevant data and context
                        data_summary = "Data not available"
                        viz_type = "unknown"
                        
                        # Look for previous dataframe to get data summary
                        for j in range(i, 0, -1):
                            if 'dataframe' in st.session_state.chat_history[j] and not st.session_state.chat_history[j]['dataframe'].empty:
                                df = st.session_state.chat_history[j]['dataframe']
                                data_summary = f"Total rows: {len(df)}, Columns: {', '.join(df.columns.tolist())}"
                                
                                # Add some descriptive statistics
                                try:
                                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                                    if len(numeric_cols) > 0:
                                        stats = df[numeric_cols].describe().to_dict()
                                        data_summary += f"\nSummary statistics: {stats}"
                                except:
                                    pass
                                
                                break
                        
                        # Determine visualization type
                        if 'fig' in st.session_state.chat_history[i]:
                            fig_type = str(type(st.session_state.chat_history[i]['fig']))
                            if 'bar' in fig_type.lower():
                                viz_type = 'bar chart'
                            elif 'line' in fig_type.lower():
                                viz_type = 'line chart'
                            elif 'pie' in fig_type.lower():
                                viz_type = 'pie chart'
                            elif 'scatter' in fig_type.lower():
                                viz_type = 'scatter plot'
                            elif 'heat' in fig_type.lower():
                                viz_type = 'heatmap'
                            
                        # Generate explanation
                        explanation = generate_viz_explanation(user_query, data_summary, viz_type)
                        
                        # Add explanation to chat
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': explanation
                        })
                        # Display explanation immediately
                        st.markdown(f"<div class='assistant-message'>{explanation}</div>", unsafe_allow_html=True)
                        break
                else:
                    # No visualization found to explain
                    response_content = "I couldn't find a chart to explain. Please create a data visualization first."
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response_content
                    })
                    # Display response immediately
                    st.markdown(f"<div class='assistant-message'>{response_content}</div>", unsafe_allow_html=True)
            
            # Process regular queries
            else:
                # Check if this is a prediction request
                prediction_keywords = ['predict', 'forecast', 'future', 'next month', 'next year', 'upcoming', 'will be']
                is_prediction_request = any(keyword in user_query.lower() for keyword in prediction_keywords)
                
                if is_prediction_request:
                    # Enhance the prediction query with advanced LLM analysis
                    enhanced_prediction_query = enhance_prediction_query(
                        user_query, 
                        st.session_state.chat_history,
                        getattr(st.session_state, 'last_dataframe', None)
                    )
                    
                    # Run the prediction function with enhanced query
                    st.markdown(f"<div class='assistant-message'>Running advanced predictive analysis...</div>", unsafe_allow_html=True)
                    success, output, fig, forecasts_df = generate_prediction(enhanced_prediction_query)
                    
                    # Debug: Print what type of figure we got back
                    print(f"Prediction returned figure of type: {type(fig)}")
                    
                    if success:
                        # Add forecast results to chat history with LLM-generated insights
                        prediction_insights = generate_prediction_insights(user_query, output, forecasts_df)
                        response_content = f"Advanced predictive analysis completed successfully.\n\n{prediction_insights}"
                        
                        # Display response immediately
                        st.markdown(f"<div class='assistant-message'>{response_content}</div>", unsafe_allow_html=True)
                        
                        # Display the forecast using enhanced render_figure function
                        try:
                            import matplotlib.figure
                            print(f"DEBUG: Figure type before rendering: {type(fig)}")
                            
                            if fig is not None and isinstance(fig, matplotlib.figure.Figure):
                                # Check if figure has any axes (simplified validation)
                                if hasattr(fig, 'get_axes') and len(fig.get_axes()) > 0:
                                    print(f"DEBUG: Figure has {len(fig.get_axes())} axes with plots - rendering")
                                    render_figure(fig)  # matplotlib figure
                                else:
                                    st.info("No visualization was generated for this prediction.")
                            elif fig is not None:
                                print(f"DEBUG: Non-matplotlib figure detected: {type(fig)}")
                                render_figure(None, fig)  # Plotly figure (or other type)
                            else:
                                print("DEBUG: No figure returned from prediction")
                                st.info("No visualization was generated for this prediction.")
                        except Exception as viz_error:
                            st.error(f"Failed to display visualization: {str(viz_error)}")
                            st.info("Please see the text analysis above for prediction results.")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response_content,
                            'fig': fig,
                            'forecasts_df': forecasts_df
                        })
                        
                        # Display raw output if it contains useful information
                        if output and len(output.strip()) > 0:
                            with st.expander("📊 Detailed Analysis Output"):
                                st.text(output)
                        
                    else:
                        # Prediction failed
                        error_message = f"Prediction analysis could not be completed: {output}"
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': error_message
                        })
                        st.markdown(f"<div class='assistant-message'>{error_message}</div>", unsafe_allow_html=True)
                else:
                    # Get query type for non-prediction queries
                    query_type = classify_query(user_query)
                    
                    # Process based on query type
                    if "sql" in query_type.lower():
                        # Generate SQL query
                        sql_query = generate_sql_query(user_query)
                        
                        # Enhance SQL if needed
                        sql_query = enhance_sql_query(user_query, sql_query)
                        
                        # Execute query
                        try:
                            print("\n📋 Database Results:")
                            result = execute_query(sql_query, user_query)
                            
                            if isinstance(result, list) and result:
                                df = pd.DataFrame([dict(row) for row in result])
                                
                                # Store the dataframe for prediction context
                                st.session_state.last_dataframe = df
                                
                                # Check if user specified a visualization type in the query
                                detected_viz_type = extract_visualization_type(user_query)
                                if detected_viz_type:
                                    viz_type = detected_viz_type
                                
                                # Add SQL query result to chat
                                response_content = generate_human_like_response(
                                    user_query, 
                                    sentiment_info, 
                                    get_conversation_context(st.session_state.chat_history),
                                    df
                                )
                                chat_message = {
                                    'role': 'assistant',
                                    'content': response_content,
                                    'sql': sql_query,
                                    'dataframe': df
                                }
                                
                                # Display response immediately
                                st.markdown(f"<div class='assistant-message'>{response_content}</div>", unsafe_allow_html=True)
                                st.code(sql_query, language="sql")
                                st.dataframe(df)
                                
                                # Add to chat history
                                st.session_state.chat_history.append(chat_message)
                                
                                # Create visualization with error handling
                                try:
                                    fig = create_visualization(df, user_query, viz_type)
                                    
                                    # Add the figure directly to the chat, don't show in a separate container
                                    if fig is not None:
                                        viz_response = "Here's the visual result of your query:"
                                        # Display visualization immediately
                                        st.markdown(f"<div class='assistant-message'>{viz_response}</div>", unsafe_allow_html=True)
                                        render_figure(None, fig)  # Use unified render_figure for Plotly charts
                                        
                                        # Show only the figure in the next message
                                        st.session_state.chat_history.append({
                                            'role': 'assistant',
                                            'content': viz_response,
                                            'fig': fig  # Now we've added the figure directly to the chat message
                                        })
                                        
                                        # Generate and add automatic explanation if the query sounds like it needs one
                                        if any(term in user_query.lower() for term in ["analyze", "show me", "explain", "what is", "how many"]):
                                            data_summary = f"Total rows: {len(df)}, Columns: {', '.join(df.columns.tolist())}"
                                            explanation = generate_viz_explanation(user_query, data_summary, viz_type)
                                            
                                            # Add context-aware explanation
                                            context_aware_explanation = f"Based on your query '{user_query}', here's what the data shows:\n\n{explanation}"
                                            
                                            st.session_state.chat_history.append({
                                                'role': 'assistant',
                                                'content': context_aware_explanation
                                            })
                                            # Display explanation immediately
                                            st.markdown(f"<div class='assistant-message'>{context_aware_explanation}</div>", unsafe_allow_html=True)
                                except Exception as viz_error:
                                    # Add error message about visualization
                                    error_message = f"An error occurred while creating the visualization: {str(viz_error)}"
                                    st.session_state.chat_history.append({
                                        'role': 'assistant',
                                        'content': error_message
                                    })
                                    st.markdown(f"<div class='assistant-message'>{error_message}</div>", unsafe_allow_html=True)
                                    st.error(error_message)
                            
                        except Exception as e:
                            # Error executing query
                            error_message = f"An error occurred while executing the SQL query: {str(e)}"
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': error_message,
                                'sql': sql_query
                            })
                            # Display error immediately
                            st.markdown(f"<div class='assistant-message'>{error_message}</div>", unsafe_allow_html=True)
                            st.code(sql_query, language="sql")
                            st.error(error_message)
                    
                    # Process company information query
                    else:
                        company_info = st.session_state.company_info
                        answer = answer_company_query(user_query, company_info)
                        
                        # Generate human-like response for company queries
                        human_response = generate_human_like_response(
                            user_query, 
                            sentiment_info, 
                            get_conversation_context(st.session_state.chat_history),
                            answer
                        )
                        
                        # Combine human greeting with company answer
                        full_response = f"{human_response}\n\n{answer}"
                        
                        # Add response to chat history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': full_response
                        })
                        
                        # Display response immediately
                        st.markdown(f"<div class='assistant-message'>{full_response}</div>", unsafe_allow_html=True)
        except Exception as e:
            # General error handling for the entire query processing
            error_message = f"An error occurred while processing your query: {str(e)}"
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': error_message
            })
            # Display error immediately
            st.markdown(f"<div class='assistant-message'>{error_message}</div>", unsafe_allow_html=True)
            st.error(error_message)

# Footer
st.markdown("---")
st.markdown("Created by: Zehra Sagin, Tugan Basaran | Text-to-SQL Project")