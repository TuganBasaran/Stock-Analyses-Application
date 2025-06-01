"""
LLM-Powered Visualization Engine
Dynamically generates visualizations based on user queries and data using AI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from io import StringIO
import sys
import re
import json
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class LLMVisualizationEngine:
    """
    AI-powered visualization engine that generates charts based on natural language queries
    """
    
    def __init__(self, model):
        self.model = model
        self.supported_types = [
            'bar', 'line', 'pie', 'scatter', 'heatmap', 'histogram', 
            'box', 'violin', 'area', 'bubble', 'treemap', 'sunburst'
        ]
    
    def analyze_query_and_data(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the user query and data to determine the best visualization approach
        """
        # Get data summary
        data_summary = self._get_data_summary(df)
        
        prompt = f"""
        Analyze this data visualization request and return a JSON response:
        
        User Query: "{query}"
        
        Data Summary:
        {data_summary}
        
        Determine:
        1. best_viz_type: Most appropriate chart type (bar, line, pie, scatter, heatmap, histogram, box, area, bubble, treemap)
        2. x_column: Best column for X-axis
        3. y_column: Best column for Y-axis  
        4. color_column: Column for color coding (if applicable)
        5. size_column: Column for sizing (if applicable)
        6. aggregation_needed: Whether data needs aggregation (true/false)
        7. aggregation_method: How to aggregate (sum, count, mean, etc.)
        8. time_series: Whether this is time-series data (true/false)
        9. title: Appropriate chart title
        10. insights: Key insights this visualization should reveal
        
        Return ONLY valid JSON with these fields.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 500,
                    "top_p": 0.9,
                }
            )
            
            analysis = json.loads(response.text.strip())
            
            # Validate and set defaults
            defaults = {
                'best_viz_type': 'bar',
                'x_column': df.columns[0] if len(df.columns) > 0 else None,
                'y_column': df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns) > 0 else df.columns[1] if len(df.columns) > 1 else None,
                'color_column': None,
                'size_column': None,
                'aggregation_needed': False,
                'aggregation_method': 'sum',
                'time_series': False,
                'title': 'Data Visualization',
                'insights': 'Data analysis results'
            }
            
            for key, default_value in defaults.items():
                if key not in analysis:
                    analysis[key] = default_value
            
            return analysis
            
        except Exception as e:
            print(f"Error in query analysis: {e}")
            return self._get_fallback_analysis(df)
    
    def generate_visualization_code(self, query: str, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """
        Generate Python code for creating the visualization
        """
        data_summary = self._get_data_summary(df)
        
        prompt = f"""
        Generate Python code to create a visualization based on this analysis:
        
        User Query: "{query}"
        Analysis: {json.dumps(analysis, indent=2)}
        
        Data Summary:
        {data_summary}
        
        Requirements:
        1. Use the DataFrame variable 'df'
        2. Create a figure and assign it to variable 'fig'
        3. Use plotly.express or plotly.graph_objects for interactive charts
        4. Handle missing data gracefully
        5. Apply proper formatting, labels, and styling
        6. Use dark theme compatible colors
        7. Include error handling for data issues
        8. Make the chart responsive and professional
        
        Available libraries: pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
        
        Return ONLY the Python code, no explanations or markdown.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 1000,
                    "top_p": 0.9,
                }
            )
            
            code = response.text.strip()
            # Clean up code blocks
            code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
            code = re.sub(r'\s*```$', '', code, flags=re.MULTILINE)
            
            return code
            
        except Exception as e:
            print(f"Error generating visualization code: {e}")
            return self._get_fallback_code(analysis)
    
    def create_visualization(self, query: str, df: pd.DataFrame) -> Tuple[Optional[Any], str, Dict[str, Any]]:
        """
        Main method to create a visualization from query and data
        """
        try:
            # Analyze query and data
            analysis = self.analyze_query_and_data(query, df)
            
            # Generate visualization code
            viz_code = self.generate_visualization_code(query, df, analysis)
            
            # Execute the code
            fig, error = self._execute_visualization_code(viz_code, df)
            
            if fig is not None:
                return fig, "Visualization created successfully", analysis
            else:
                return None, f"Error creating visualization: {error}", analysis
                
        except Exception as e:
            return None, f"Error in visualization creation: {str(e)}", {}
    
    def create_prediction_visualization(self, query: str, historical_data: pd.DataFrame, 
                                     forecast_data: pd.DataFrame = None) -> Tuple[Optional[Any], str]:
        """
        Create specialized visualizations for prediction/forecasting
        """
        prompt = f"""
        Generate Python code for a prediction/forecasting visualization:
        
        User Query: "{query}"
        Historical Data Columns: {list(historical_data.columns)}
        Historical Data Shape: {historical_data.shape}
        Forecast Data Available: {forecast_data is not None}
        
        Create a professional prediction chart that:
        1. Shows historical data as a line/bar chart
        2. Shows forecast data in a different style (dashed line, different color)
        3. Clearly separates historical from predicted values
        4. Includes confidence intervals if possible
        5. Uses appropriate time-based x-axis
        6. Has professional styling for business presentations
        7. Includes trend lines and key insights
        
        Use plotly for interactive charts. Assign result to 'fig' variable.
        Available data: 'historical_data' and 'forecast_data' DataFrames.
        
        Return ONLY Python code.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 1200,
                    "top_p": 0.9,
                }
            )
            
            code = response.text.strip()
            code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
            code = re.sub(r'\s*```$', '', code, flags=re.MULTILINE)
            
            # Execute the prediction visualization code
            local_vars = {
                'historical_data': historical_data.copy(),
                'forecast_data': forecast_data.copy() if forecast_data is not None else pd.DataFrame(),
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
                'datetime': datetime,
                'timedelta': timedelta
            }
            
            fig, error = self._execute_code_safely(code, local_vars)
            
            if fig is not None:
                return fig, "Prediction visualization created successfully"
            else:
                return None, f"Error creating prediction visualization: {error}"
                
        except Exception as e:
            return None, f"Error in prediction visualization: {str(e)}"
    
    def _get_data_summary(self, df: pd.DataFrame) -> str:
        """Get a comprehensive summary of the DataFrame"""
        try:
            summary = f"""
            Shape: {df.shape}
            Columns: {list(df.columns)}
            Data Types: {df.dtypes.to_dict()}
            """
            
            # Add sample data
            if len(df) > 0:
                summary += f"\nSample Data:\n{df.head(3).to_string()}"
            
            # Add numeric column statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary += f"\nNumeric Columns: {list(numeric_cols)}"
                summary += f"\nNumeric Summary:\n{df[numeric_cols].describe().to_string()}"
            
            # Add categorical column info
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                summary += f"\nCategorical Columns: {list(categorical_cols)}"
                for col in categorical_cols[:3]:  # Limit to first 3
                    unique_vals = df[col].nunique()
                    summary += f"\n{col}: {unique_vals} unique values"
            
            return summary
            
        except Exception as e:
            return f"Error getting data summary: {str(e)}"
    
    def _get_fallback_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        return {
            'best_viz_type': 'bar' if len(categorical_cols) > 0 else 'line',
            'x_column': categorical_cols[0] if len(categorical_cols) > 0 else df.columns[0],
            'y_column': numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1] if len(df.columns) > 1 else df.columns[0],
            'color_column': None,
            'size_column': None,
            'aggregation_needed': False,
            'aggregation_method': 'sum',
            'time_series': any('date' in col.lower() for col in df.columns),
            'title': 'Data Visualization',
            'insights': 'Basic data visualization'
        }
    
    def _get_fallback_code(self, analysis: Dict[str, Any]) -> str:
        """Generate fallback visualization code"""
        viz_type = analysis.get('best_viz_type', 'bar')
        x_col = analysis.get('x_column', 'x')
        y_col = analysis.get('y_column', 'y')
        title = analysis.get('title', 'Visualization')
        
        if viz_type == 'bar':
            return f"""
import plotly.express as px
fig = px.bar(df, x='{x_col}', y='{y_col}', title='{title}')
fig.update_layout(template='plotly_dark')
"""
        elif viz_type == 'line':
            return f"""
import plotly.express as px
fig = px.line(df, x='{x_col}', y='{y_col}', title='{title}')
fig.update_layout(template='plotly_dark')
"""
        else:
            return f"""
import plotly.express as px
fig = px.scatter(df, x='{x_col}', y='{y_col}', title='{title}')
fig.update_layout(template='plotly_dark')
"""
    
    def _execute_visualization_code(self, code: str, df: pd.DataFrame) -> Tuple[Optional[Any], Optional[str]]:
        """Execute visualization code safely"""
        local_vars = {
            'df': df.copy(),
            'pd': pd,
            'np': np,
            'px': px,
            'go': go,
            'plt': plt,
            'sns': sns
        }
        
        return self._execute_code_safely(code, local_vars)
    
    def _execute_code_safely(self, code: str, local_vars: Dict) -> Tuple[Optional[Any], Optional[str]]:
        """Safely execute code and return figure"""
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            exec(code, {}, local_vars)
            sys.stdout = original_stdout
            
            fig = local_vars.get('fig', None)
            if fig is not None:
                return fig, None
            else:
                return None, "No figure variable created"
                
        except Exception as e:
            sys.stdout = original_stdout
            return None, str(e)
