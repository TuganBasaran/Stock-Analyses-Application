import sqlalchemy as db
from sqlalchemy.sql import text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
from sqlalchemy import Enum
import enum 
import json
from pathlib import Path

'''
CODE AÇIKLAMASI
---------------
1- Mysql üzerinden yeni bir database yarat. Adını 'AI_390' koy
2- mysql+pymysql://{mysql_kullanıcı_adı}:{mysql_kullanıcı_şifren}@localhost/{database_adın}
3- Bu dosyayı bir kez run'layınca otomatik olarak table'ın AI_390 database'i içerisinde oluşturulacak 
4- run_once.py dosyasına geç 
5- Bir kez daha run'lamaya gerek kalmadı. 

VERİ KALİTESİ NOTLARI:
- JSON verisinde duplicate order_id'ler mevcut (aynı order_id farklı tarihlerde)
- Bu gerçek dünya veri kalitesi sorununu yansıtıyor
- Orders tablosunda id field'ı primary key, order_id field'ı ise business identifier
'''

# MySQL database configuration - direct approach
import os
from pathlib import Path
from dotenv import load_dotenv

# Database connection configurations
DB_CONFIG = {
    'user': 'root',
    'password': 'password',  # Default password
    'host': 'localhost',
    'database': 'AI_390',
}

# Try to load environment variables if .env exists
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
    
    # Override defaults with environment variables if they exist
    if os.getenv('DB_USER'):
        DB_CONFIG['user'] = os.getenv('DB_USER')
    if os.getenv('DB_PASSWORD'):
        DB_CONFIG['password'] = os.getenv('DB_PASSWORD')
    if os.getenv('DB_HOST'):
        DB_CONFIG['host'] = os.getenv('DB_HOST')
    if os.getenv('DB_NAME'):
        DB_CONFIG['database'] = os.getenv('DB_NAME')

# Create MySQL connection string
connection_string = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
print(f"Connecting to MySQL database: {DB_CONFIG['host']}/{DB_CONFIG['database']}")

# Create engine with error handling
try:
    engine = db.create_engine(connection_string)
    # Test connection
    connection = engine.connect()
    connection.close()
    print("✅ Database connection successful")
except Exception as e:
    print(f"❌ Database connection error: {str(e)}")
    print("⚠️ Using default engine configuration")
    # Fallback to a minimal connection for development
    engine = db.create_engine('mysql+pymysql://root:password@localhost/AI_390')

Base = declarative_base()

# PRODUCTS TABLE 

class Products(Base): 
    __tablename__= 'products'  # Tablo adını küçük harfle değiştirdik

    brand_name = db.Column(db.String(255), nullable= False)
    category_name = db.Column(db.String(255), nullable= False)
    product_id= db.Column(db.String(255), primary_key= True)
    release_date = db.Column(db.Integer, nullable= False)  # Sadece yıl değeri olarak saklanacak

class Status(enum.Enum): 
    created= 'Created'
    cancelled= 'Cancelled'
    # Note: 'Returned' is not used in the current dataset

# ORDERS TABLE 

class Orders(Base): 
    __tablename__ = 'orders'  # Tablo adı küçük harflerle olmalı

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Otomatik artan birincil anahtar
    customer_id = db.Column(db.String(255), nullable= False) 
    location = db.Column(db.String(255), nullable= False)
    order_date = db.Column(db.DateTime(timezone= True), server_default= db.func.now())
    order_id = db.Column(db.String(255), nullable= False)  # Artık primary key değil
    price = db.Column(db.Float, nullable= False)
    product_id= db.Column(db.String(255), db.ForeignKey('products.product_id'), nullable= False)
    seller_id= db.Column(db.String(255), nullable= False)
    status= db.Column(db.Enum(Status))
    
    # Relationship to Products table
    product = relationship("Products", backref="orders")

# Sorgu geçmişi tablosu
class QueryHistory(Base):
    __tablename__ = 'query_history'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    query = db.Column(db.Text, nullable=False)
    user_input = db.Column(db.Text, nullable=True)
    execution_time = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"ID: {self.id} | Query: {self.query[:50]}... | Time: {self.execution_time}"

# Tabloları Oluşturma 
# Mevcut veritabanını korumak için tabloları oluşturma kodunu yorumlayalım
# Base.metadata.create_all(engine)

Session = sessionmaker(bind= engine)
session = Session()

def add_order(customer_id, location, price, product_id, seller_id, status, order_date= None): 
    order= Orders(
        customer_id= customer_id,
        location= location,
        price= price, 
        product_id= product_id, 
        seller_id= seller_id,
        status= status, 
        order_date= order_date or datetime.now()
    )
    session.add(order)
    session.commit()
    return order.order_id

def delete_all(): 
    deleted_count = session.query(Orders).delete()
    session.execute(text("TRUNCATE TABLE Orders"))
    session.commit()
    return deleted_count

# Tabloları yeniden oluşturmak için yardımcı fonksiyon
def recreate_tables():
    """Tüm tabloları silip yeniden oluşturur - DİKKAT: Tüm veriler silinir!"""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return "Tablolar yeniden oluşturuldu"

def load_products_from_json():
    """Products.json dosyasından verileri bulk insert ile hızlı yükler"""
    try:
        products_path = Path("data/json/products.json")
        if not products_path.exists():
            print(f"Products.json file not found: {products_path}")
            return False
        
        print("🔄 Reading products data...")
        products_to_insert = []
        
        with open(products_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    product = json.loads(line.strip())
                    products_to_insert.append({
                        'product_id': product.get('productid'),
                        'category_name': product.get('categoryname'),
                        'brand_name': product.get('brandname'),
                        'release_date': product.get('releasedate')
                    })
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Product parsing error: {str(e)}")
                    continue
        
        print(f"📦 Inserting {len(products_to_insert)} products using bulk insert...")
        
        # Clear existing products first
        session.query(Products).delete()
        
        # Bulk insert - much faster!
        session.bulk_insert_mappings(Products, products_to_insert)
        session.commit()
        
        print(f"✅ {len(products_to_insert)} products successfully loaded to database")
        return True
    except Exception as e:
        session.rollback()
        print(f"❌ Products loading error: {str(e)}")
        return False

def add_to_history(query, user_input=None):
    """SQL sorgusunu ve kullanıcı girdisini history tablosuna kaydeder"""
    history_entry = QueryHistory(
        query=query,
        user_input=user_input
    )
    session.add(history_entry)
    session.commit()
    return history_entry.id

def get_query_history(limit=20):
    """En son çalıştırılan sorguları listeler"""
    history = session.query(QueryHistory).order_by(QueryHistory.execution_time.desc()).limit(limit).all()
    for entry in history:
        # Show full query without truncation
        print(f"ID: {entry.id} | Time: {entry.execution_time}")
        print(f"Query: {entry.query}")
        # Add a separator for better readability
        print("-" * 50)
    return history

def clear_query_history():
    """Tüm sorgu geçmişini temizler"""
    deleted_count = session.query(QueryHistory).delete()
    session.commit()
    return deleted_count

def execute_query(query, user_input=None):
    '''Raw SQL query executer method'''
    # Sorguyu history'ye ekle
    if user_input:
        add_to_history(query, user_input)
    else:
        add_to_history(query)
    
    result = session.execute(text(query))
    
    # Get column names from the result
    column_names = result.keys()
    
    # Format and print results
    rows = []
    for row in result:
        # Her satır için doğru bir sözlük oluştur
        row_dict = {}
        for idx, col_name in enumerate(column_names):
            row_dict[col_name] = row[idx]
            
        # Satırı ekrana yazdır (debugging için)
        row_data = []
        for idx, col_name in enumerate(column_names):
            row_data.append(f"{col_name}: {row[idx]}")
        print(" | ".join(row_data))
        
        # Sözlüğü rows listesine ekle
        rows.append(row_dict)
    
    return rows

def execute_raw_query(query):
    """Execute raw SQL query and return results as list of dictionaries without logging to history"""
    try:
        result = session.execute(text(query))
        
        # Get column names from the result
        column_names = result.keys()
        
        # Format results as list of dictionaries
        rows = []
        for row in result:
            row_dict = {}
            for idx, col_name in enumerate(column_names):
                row_dict[col_name] = row[idx]
            rows.append(row_dict)
        
        return rows
    except Exception as e:
        print(f"Database query error: {str(e)}")
        return None