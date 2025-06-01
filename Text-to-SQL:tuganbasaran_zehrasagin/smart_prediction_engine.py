"""
Smart Prediction Engine using LLM for flexible forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class SmartPredictionEngine:
    """
    LLM-powered prediction engine that generates forecasts based on natural language queries
    """
    
    def __init__(self, model):
        self.model = model
    
    def analyze_prediction_request(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the prediction request to understand what the user wants
        """
        data_summary = self._get_data_summary(data)
        
        prompt = f"""
        Analyze this prediction request and return JSON:
        
        User Query: "{query}"
        
        Data Summary:
        {data_summary}
        
        Determine:
        1. prediction_type: Type of prediction (revenue, sales, growth, trend, seasonal)
        2. time_horizon: How far to predict (days, weeks, months, quarters, years)
        3. time_periods: Number of periods to predict (1, 3, 6, 12, etc.)
        4. target_column: Column to predict
        5. date_column: Column containing dates
        6. aggregation_level: How to aggregate data (daily, weekly, monthly, quarterly)
        7. seasonality_expected: Whether seasonal patterns are expected (true/false)
        8. confidence_intervals: Whether to include confidence intervals (true/false)
        9. business_context: Business context for the prediction
        10. key_metrics: Key metrics to calculate and display
        
        Return ONLY valid JSON.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 400,
                    "top_p": 0.9,
                }
            )
            
            analysis = json.loads(response.text.strip())
            
            # Set defaults for missing fields
            defaults = {
                'prediction_type': 'revenue',
                'time_horizon': 'months',
                'time_periods': 6,
                'target_column': self._find_target_column(data),
                'date_column': self._find_date_column(data),
                'aggregation_level': 'monthly',
                'seasonality_expected': True,
                'confidence_intervals': True,
                'business_context': 'Business forecasting',
                'key_metrics': ['forecast_value', 'growth_rate', 'trend_direction']
            }
            
            for key, default_value in defaults.items():
                if key not in analysis:
                    analysis[key] = default_value
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing prediction request: {e}")
            return self._get_fallback_prediction_analysis(data)
    
    def generate_prediction_code(self, query: str, data: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """
        Generate Python code for prediction analysis
        """
        prompt = f"""
        Generate Python code for prediction analysis:
        
        User Query: "{query}"
        Analysis: {json.dumps(analysis, indent=2)}
        
        Data columns: {list(data.columns)}
        Data shape: {data.shape}
        
        Create code that:
        1. Prepares and aggregates the data appropriately
        2. Generates forecasts using simple but effective methods
        3. Calculates confidence intervals
        4. Computes key business metrics
        5. Creates a results DataFrame with predictions
        6. Handles missing data and edge cases
        7. Returns both numerical results and insights
        
        Use only pandas, numpy, and basic statistical methods.
        Assign final results to 'forecast_results' and 'insights_text' variables.
        
        Return ONLY Python code, no explanations.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 1500,
                    "top_p": 0.9,
                }
            )
            
            code = response.text.strip()
            code = code.replace('```python', '').replace('```', '').strip()
            
            return code
            
        except Exception as e:
            print(f"Error generating prediction code: {e}")
            return self._get_fallback_prediction_code(analysis)
    
    def create_prediction(self, query: str, data: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Main method to create predictions
        """
        try:
            # Analyze the prediction request
            analysis = self.analyze_prediction_request(query, data)
            
            # Generate prediction code
            prediction_code = self.generate_prediction_code(query, data, analysis)
            
            # Execute the prediction
            success, results, insights = self._execute_prediction_code(prediction_code, data)
            
            if success:
                return True, insights, results, analysis
            else:
                return False, f"Prediction failed: {insights}", None, analysis
                
        except Exception as e:
            return False, f"Error in prediction creation: {str(e)}", None, {}
    
    def generate_business_insights(self, query: str, forecast_results: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """
        Generate business insights from prediction results
        """
        if forecast_results is None or forecast_results.empty:
            return "No forecast data available for insights generation."
        
        results_summary = forecast_results.to_string()
        
        prompt = f"""
        Generate business insights from these prediction results:
        
        Original Query: "{query}"
        Prediction Analysis: {json.dumps(analysis, indent=2)}
        
        Forecast Results:
        {results_summary}
        
        Provide:
        1. Executive Summary (2-3 sentences)
        2. Key Findings (3-4 bullet points)
        3. Business Implications (2-3 bullet points)
        4. Recommendations (2-3 actionable items)
        5. Risk Factors (1-2 potential concerns)
        
        Write in clear, professional business language.
        Focus on actionable insights and strategic implications.
        Keep the response concise but comprehensive.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.4,
                    "max_output_tokens": 600,
                    "top_p": 0.9,
                }
            )
            
            return response.text.strip()
            
        except Exception as e:
            return f"Error generating business insights: {str(e)}"
    
    def _get_data_summary(self, data: pd.DataFrame) -> str:
        """Get summary of the data for analysis"""
        try:
            summary = f"""
            Shape: {data.shape}
            Columns: {list(data.columns)}
            Date columns: {[col for col in data.columns if 'date' in col.lower()]}
            Numeric columns: {list(data.select_dtypes(include=[np.number]).columns)}
            """
            
            if len(data) > 0:
                summary += f"\nDate range: {data.select_dtypes(include=['datetime64']).min().min() if len(data.select_dtypes(include=['datetime64']).columns) > 0 else 'No dates'} to {data.select_dtypes(include=['datetime64']).max().max() if len(data.select_dtypes(include=['datetime64']).columns) > 0 else 'No dates'}"
            
            return summary
            
        except Exception as e:
            return f"Error getting data summary: {str(e)}"
    
    def _find_target_column(self, data: pd.DataFrame) -> str:
        """Find the most likely target column for prediction"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Look for common target column names
        target_keywords = ['amount', 'price', 'revenue', 'sales', 'value', 'total', 'cost']
        
        for keyword in target_keywords:
            for col in numeric_cols:
                if keyword in col.lower():
                    return col
        
        # Return first numeric column if no keyword match
        return numeric_cols[0] if len(numeric_cols) > 0 else data.columns[0]
    
    def _find_date_column(self, data: pd.DataFrame) -> str:
        """Find the date column in the data"""
        date_keywords = ['date', 'time', 'created', 'updated']
        
        for keyword in date_keywords:
            for col in data.columns:
                if keyword in col.lower():
                    return col
        
        # Check for datetime columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        return datetime_cols[0] if len(datetime_cols) > 0 else data.columns[0]
    
    def _get_fallback_prediction_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        return {
            'prediction_type': 'revenue',
            'time_horizon': 'months',
            'time_periods': 6,
            'target_column': self._find_target_column(data),
            'date_column': self._find_date_column(data),
            'aggregation_level': 'monthly',
            'seasonality_expected': True,
            'confidence_intervals': True,
            'business_context': 'Business forecasting',
            'key_metrics': ['forecast_value', 'growth_rate', 'trend_direction']
        }
    
    def _get_fallback_prediction_code(self, analysis: Dict[str, Any]) -> str:
        """Generate fallback prediction code"""
        target_col = analysis.get('target_column', 'amount')
        date_col = analysis.get('date_column', 'order_date')
        periods = analysis.get('time_periods', 6)
        
        return f"""
# Simple prediction using moving average and trend
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    # Prepare data
    data['{date_col}'] = pd.to_datetime(data['{date_col}'])
    monthly_data = data.groupby(pd.Grouper(key='{date_col}', freq='ME'))['{target_col}'].sum()
    monthly_data = monthly_data[monthly_data > 0]
    
    # Simple forecasting
    recent_avg = monthly_data.tail(3).mean()
    growth_rate = monthly_data.pct_change().tail(3).mean()
    
    # Generate forecasts
    forecasts = []
    for i in range({periods}):
        forecast_value = recent_avg * (1 + growth_rate) ** (i + 1)
        forecast_date = monthly_data.index[-1] + pd.DateOffset(months=i+1)
        forecasts.append({{'date': forecast_date, 'forecast': forecast_value}})
    
    forecast_results = pd.DataFrame(forecasts)
    
    insights_text = f"Forecast generated using moving average method. Expected growth rate: {{growth_rate*100:.1f}}% per month."
    
except Exception as e:
    forecast_results = pd.DataFrame()
    insights_text = f"Error in prediction: {{str(e)}}"
"""
    
    def _execute_prediction_code(self, code: str, data: pd.DataFrame) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """Execute prediction code safely"""
        local_vars = {
            'data': data.copy(),
            'pd': pd,
            'np': np,
            'datetime': datetime,
            'timedelta': timedelta,
            'forecast_results': pd.DataFrame(),
            'insights_text': 'No insights generated'
        }
        
        try:
            exec(code, {}, local_vars)
            
            forecast_results = local_vars.get('forecast_results', pd.DataFrame())
            insights_text = local_vars.get('insights_text', 'Prediction completed')
            
            return True, forecast_results, insights_text
            
        except Exception as e:
            return False, None, f"Error executing prediction code: {str(e)}"
