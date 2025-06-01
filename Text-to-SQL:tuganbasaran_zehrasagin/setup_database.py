#!/usr/bin/env python3
"""
Loads Products data into database and establishes foreign key relationships
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import db

def main():
    print("🔧 Loading Products data into database...")
    
    # First recreate tables (for foreign key relationships)
    try:
        print("📋 Recreating tables...")
        result = db.recreate_tables()
        print(f"✅ {result}")
    except Exception as e:
        print(f"❌ Table creation error: {str(e)}")
        return False
    
    # Load Products data
    try:
        print("📦 Loading Products data from JSON...")
        success = db.load_products_from_json()
        if success:
            print("✅ Products data loaded successfully!")
        else:
            print("❌ Products data could not be loaded!")
            return False
    except Exception as e:
        print(f"❌ Products loading error: {str(e)}")
        return False
    
    # Reload Orders data by calling run_once.py
    try:
        print("📊 Reloading Orders data...")
        os.system("cd db && python run_once.py")
        print("✅ Database setup completed!")
    except Exception as e:
        print(f"❌ Orders loading error: {str(e)}")
        return False
    
    # Test JOIN query
    try:
        print("\n🧪 Testing JOIN query...")
        test_query = """
        SELECT o.price, p.category_name, p.brand_name 
        FROM orders o 
        JOIN products p ON o.product_id = p.product_id 
        LIMIT 5
        """
        result = db.execute_raw_query(test_query)
        if result:
            print(f"✅ JOIN query successful! {len(result)} rows returned")
            for row in result:
                print(f"  Price: ${row['price']}, Category: {row['category_name']}, Brand: {row['brand_name']}")
        else:
            print("❌ JOIN query failed!")
    except Exception as e:
        print(f"❌ JOIN test error: {str(e)}")
    
    return True

if __name__ == "__main__":
    main()
