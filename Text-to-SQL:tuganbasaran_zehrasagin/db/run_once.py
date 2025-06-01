from db import session, Orders, Products, Status, recreate_tables
from datetime import datetime
from pathlib import Path
import json

''' UYARI: BU KISMI SADECE BİR KERE RUNLA!!!!!
    YOKSA AYNI DATADAN BİRDEN FAZLA OLACAK 

    CODE AÇIKLAMASI 
    --------------
    1- Database'i oluşturduktan sonra json file'ı read'lemen lazım 
    2- Her json satırını parse edip database'e ekliyorsun
    3- Detaylı açıklama aşağıda
'''

# 1. JSON yollarını dinamik şekilde ayarla
orders_json_path = Path(__file__).resolve().parent.parent / "data" / "json" / "orders.json"
products_json_path = Path(__file__).resolve().parent.parent / "data" / "json" / "products.json"

# 2. Veritabanını sıfırla ve tabloları yeniden oluştur
print("Recreating database tables...")
recreate_tables()
print("Tables recreated successfully!")

# 3. Önce products.json dosyasını işleyelim
product_count = 0

print("Importing products...")
with open(products_json_path, 'r') as file:
    for line in file:
        try:
            product_data = json.loads(line)
            
            # release_date'i direkt integer olarak al
            release_year = product_data.get('releasedate')
            if release_year:
                # Sayısal değere çevir
                if isinstance(release_year, int):
                    # Zaten integer
                    release_date = release_year
                elif isinstance(release_year, str):
                    # String ise sayıya çevir
                    if release_year.isdigit():
                        release_date = int(release_year)
                    else:
                        # Eğer tarih formatındaysa yılını al
                        try:
                            date_obj = datetime.fromisoformat(release_year)
                            release_date = date_obj.year
                        except ValueError:
                            print(f"Warning: Invalid release date format: {release_year} for product {product_data.get('productid')}")
                            release_date = 2020  # Varsayılan değer
                else:
                    release_date = 2020  # Varsayılan değer
            else:
                # Tarih yoksa varsayılan değer
                release_date = 2020
            
            # Yeni bir ürün oluştur - Products tablosu alanlarına göre ayarla
            product = Products(
                product_id=product_data.get('productid'),
                brand_name=product_data.get('brandname'),
                category_name=product_data.get('categoryname'),
                release_date=release_date
            )
            
            # Veritabanına ekle
            session.add(product)
            product_count += 1
            
            # Her 100 kayıtta bir commit yap (performans için)
            if product_count % 100 == 0:
                session.commit()
                print(f"{product_count} products imported so far...")
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in products: {e}")
        except Exception as e:
            print(f"Error processing product line: {e}")
            print(f"Problematic data: {line}")
            session.rollback()

# Son kalan kayıtları commit et
session.commit()
print(f'{product_count} products imported successfully!')

# 4. Şimdi orders.json dosyasını işleyelim
order_count = 0

print("Importing orders...")
with open(orders_json_path, 'r') as file:
    for line in file:
        # JSON satırını parse et
        try:
            order_data = json.loads(line)
            
            # Tarih bilgisini parse et (ISO format: 2023-09-28T07:49:23.000000)
            order_date_str = order_data.get('order_date')
            order_date = datetime.fromisoformat(order_date_str) if order_date_str else None
            
            # Status değerini doğru enum değerine çevir
            status_str = order_data.get('status')
            if status_str == 'Created':
                status_value = Status.created
            elif status_str == 'Cancelled':
                status_value = Status.cancelled
            else:
                status_value = None
            
            # Orders tablosunun alanlarına göre yeni sipariş nesnesi oluştur
            order = Orders(
                customer_id=order_data.get('customer_id'),
                location=order_data.get('location'),
                price=float(order_data.get('price', 0)),
                product_id=order_data.get('product_id'),
                seller_id=order_data.get('seller_id'),
                status=status_value,
                order_date=order_date,
                order_id=order_data.get('order_id')  # order_id'yi doğrudan burada atayabiliriz
            )
            
            # Veritabanına ekle
            session.add(order)
            order_count += 1
            
            # Her 100 kayıtta bir commit yap (performans için)
            if order_count % 100 == 0:
                session.commit()
                print(f"{order_count} orders imported so far...")
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in orders: {e}")
        except Exception as e:
            print(f"Error processing order line: {e}")
            print(f"Problematic JSON: {line}")
            session.rollback()

# Son kalan kayıtları commit et
session.commit()
print(f'{order_count} orders imported successfully!')

# 5. Veri doğrulama
all_products = session.query(Products).all()
all_orders = session.query(Orders).all()

print(f"Total products in database: {len(all_products)}")
print(f"Total orders in database: {len(all_orders)}")

# İlk 5 ürünü göster
print("\nSample Products:")
for product in all_products[:5]:
    print(f"ID: {product.product_id} | Brand: {product.brand_name} | Category: {product.category_name}")

# İlk 5 siparişi göster
print("\nSample Orders:")
for order in all_orders[:5]:
    print(f"ID: {order.order_id} | Customer: {order.customer_id} | Date: {order.order_date}")
    print(f"Location: {order.location} | Price: {order.price} | Product: {order.product_id}")
    print(f"Seller: {order.seller_id} | Status: {order.status}")
    print("-" * 50)

# 6. İlişki doğrulama - Örnek bir ilişkili sorgu
print("\nSample Joined Query:")
try:
    # Orders ve Products tablolarını product_id üzerinden birleştir
    joined_result = session.query(Orders, Products).join(
        Products, Orders.product_id == Products.product_id
    ).limit(5).all()
    
    for order, product in joined_result:
        print(f"Order ID: {order.order_id} | Customer: {order.customer_id}")
        print(f"Product: {product.brand_name} - {product.category_name}")
        print("-" * 50)
except Exception as e:
    print(f"Error during join query: {e}")

print("\nData import completed successfully!")