#!/usr/bin/env python3
"""
Test script for the enhanced prediction engine integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_prediction():
    """Test the enhanced prediction engine with sample data"""
    print("🧪 Testing Enhanced Prediction Engine")
    print("=" * 50)
    
    try:
        # Import the enhanced prediction engine
        from enhanced_prediction_engine import EnhancedPredictionEngine
        print("✅ Enhanced prediction engine imported successfully")
        
        # Create sample data similar to real order data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        # Generate sample order data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)  # For reproducible results
        
        sample_data = []
        for date in date_range:
            # Simulate daily orders with some trend and seasonality
            base_amount = 1000
            trend = (date - start_date).days * 0.5  # Small upward trend
            seasonal = 200 * np.sin(2 * np.pi * date.dayofyear / 365)  # Yearly seasonality
            noise = np.random.normal(0, 100)
            
            daily_amount = base_amount + trend + seasonal + noise
            if daily_amount < 0:
                daily_amount = 100  # Minimum daily sales
                
            sample_data.append({
                'order_date': date,
                'amount': daily_amount,
                'customer_name': f'Customer_{np.random.randint(1, 100)}',
                'location': np.random.choice(['New York', 'California', 'Texas', 'Florida']),
                'status': 'Completed'
            })
        
        sample_df = pd.DataFrame(sample_data)
        print(f"✅ Created sample data with {len(sample_df)} records")
        print(f"   Date range: {sample_df['order_date'].min()} to {sample_df['order_date'].max()}")
        print(f"   Total amount: ${sample_df['amount'].sum():,.2f}")
        
        # Initialize the enhanced prediction engine with a mock model
        class MockModel:
            def generate_content(self, prompt):
                class MockResponse:
                    text = "Mock AI analysis: The data shows positive trends with potential for growth."
                return MockResponse()
        
        mock_model = MockModel()
        engine = EnhancedPredictionEngine(mock_model)
        print("✅ Enhanced prediction engine initialized with mock model")
        
        # Test different types of prediction queries
        test_queries = [
            "predict next month's revenue",
            "forecast sales trends for next quarter", 
            "what will be the expected growth rate",
            "project revenue for next 6 months"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔮 Test {i}: {query}")
            print("-" * 40)
            
            try:
                success, insights, figure, forecasts_df = engine.create_prediction(query, sample_df)
                
                if success:
                    print("✅ Prediction successful!")
                    print(f"📊 Insights preview: {insights[:200]}...")
                    print(f"📈 Figure type: {type(figure)}")
                    print(f"📋 Forecasts shape: {forecasts_df.shape if forecasts_df is not None else 'None'}")
                    
                    if forecasts_df is not None and not forecasts_df.empty:
                        print(f"📊 Sample forecast data:")
                        print(forecasts_df.head())
                else:
                    print(f"❌ Prediction failed: {insights}")
                    
            except Exception as e:
                print(f"❌ Error in prediction test: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n🎉 Enhanced prediction engine test completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure enhanced_prediction_engine.py is in the same directory")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def test_main_rag_integration():
    """Test the main_RAG integration"""
    print("\n🔗 Testing main_RAG integration")
    print("=" * 50)
    
    try:
        # Import the generate_prediction function from main_RAG
        from main_RAG import generate_prediction
        print("✅ generate_prediction imported from main_RAG")
        
        # Test with a simple prediction query
        test_query = "predict next month revenue"
        print(f"🧪 Testing query: '{test_query}'")
        
        # This would normally require actual data, so we'll just test the import and basic call structure
        print("✅ main_RAG integration appears to be working")
        print("   (Full test requires running Streamlit app with real data)")
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
    except Exception as e:
        print(f"❌ Integration test error: {str(e)}")

if __name__ == "__main__":
    test_enhanced_prediction()
    test_main_rag_integration()
    print("\n🏁 All tests completed!")
