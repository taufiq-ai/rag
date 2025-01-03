
company_details = """# TechVision Electronics

## About Us
Founded 2005 in Silicon Valley. Leading online retailer of gaming/tech equipment. "Excellence in E-commerce" award winner 2021-2023.

## Legacy
- Started: 5 employees, 100 products
- Now: 500+ team, 50,000+ products 
- Innovations: Same-day delivery, AR product previews

## Core Business
- Premium gaming peripherals
- Custom PC builds
- High-end electronics
- 5 physical experience centers
- 60-day returns
- Price-match guarantee

## Contact
- Support: 1-800-TECHVISION (24/7)
- Business: partners@techvision.com
- Press: media@techvision.com
- Careers: careers@techvision.com

## Locations
**HQ:** 1234 Innovation Drive, San Jose, CA 95110

**Experience Centers:**
- Silicon Valley Mall, CA
- Times Square, NY
- Michigan Ave, Chicago
- Downtown Seattle, WA
- Aventura Mall, Miami

## Support
- Phone: 1-888-TECH-HELP
- Chat: 24/7 on website
- Email: support@techvision.com
- Response: < 2 hours

## Impact
- Green Gaming initiative
- Carbon-neutral shipping
- TechEd Foundation
- STEM education support
"""

products = [
        {
            "id": 100001,
            "product_name": "Product-100001",
            "brand": "Razer",
            "price": 50.0,
            "discount_percent": 5.0,
            "discounted_price": 47.5,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Gaming Peripherals",
            "sub_category": "Sub-Gaming Peripherals",
            "description": "High-quality Gaming Peripherals product with premium features",
            "specifications": {
                "weight": "0.5kg",
                "dimensions": "20x15x5cm",
                "color": "Black"
            },
            "ratings": 3.5,
            "reviews_count": 13,
            "warranty_months": 12,
            "added_date": "2024-11-30",
            "tags": [
                "Gaming Peripherals",
                "Razer",
                "Premium"
            ]
        },
        {
            "id": 100002,
            "product_name": "Product-100002",
            "brand": "Logitech",
            "price": 93.75,
            "discount_percent": 6.0,
            "discounted_price": 88.12,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Monitors",
            "sub_category": "Sub-Monitors",
            "description": "High-quality Monitors product with premium features",
            "specifications": {
                "weight": "0.6kg",
                "dimensions": "21x16x6cm",
                "color": "White"
            },
            "ratings": 3.6,
            "reviews_count": 26,
            "warranty_months": 24,
            "added_date": "2024-11-25",
            "tags": [
                "Monitors",
                "Logitech",
                "Budget"
            ]
        },
        {
            "id": 100003,
            "product_name": "Product-100003",
            "brand": "Corsair",
            "price": 137.5,
            "discount_percent": 7.0,
            "discounted_price": 127.87,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Laptops",
            "sub_category": "Sub-Laptops",
            "description": "High-quality Laptops product with premium features",
            "specifications": {
                "weight": "0.7kg",
                "dimensions": "22x17x7cm",
                "color": "Gray"
            },
            "ratings": 3.7,
            "reviews_count": 39,
            "warranty_months": 36,
            "added_date": "2024-11-20",
            "tags": [
                "Laptops",
                "Corsair",
                "Mid-range"
            ]
        },
        {
            "id": 100004,
            "product_name": "Product-100004",
            "brand": "ASUS",
            "price": 181.25,
            "discount_percent": 8.0,
            "discounted_price": 166.75,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Components",
            "sub_category": "Sub-Components",
            "description": "High-quality Components product with premium features",
            "specifications": {
                "weight": "0.8kg",
                "dimensions": "23x18x8cm",
                "color": "Black"
            },
            "ratings": 3.8,
            "reviews_count": 52,
            "warranty_months": 12,
            "added_date": "2024-11-15",
            "tags": [
                "Components",
                "ASUS",
                "Premium"
            ]
        },
        {
            "id": 100005,
            "product_name": "Product-100005",
            "brand": "Dell",
            "price": 225.0,
            "discount_percent": 9.0,
            "discounted_price": 204.75,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Storage",
            "sub_category": "Sub-Storage",
            "description": "High-quality Storage product with premium features",
            "specifications": {
                "weight": "0.9kg",
                "dimensions": "24x19x9cm",
                "color": "White"
            },
            "ratings": 3.9,
            "reviews_count": 65,
            "warranty_months": 24,
            "added_date": "2024-11-10",
            "tags": [
                "Storage",
                "Dell",
                "Budget"
            ]
        },
        {
            "id": 100006,
            "product_name": "Product-100006",
            "brand": "Samsung",
            "price": 268.75,
            "discount_percent": 10.0,
            "discounted_price": 241.88,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Audio",
            "sub_category": "Sub-Audio",
            "description": "High-quality Audio product with premium features",
            "specifications": {
                "weight": "1.0kg",
                "dimensions": "25x20x10cm",
                "color": "Gray"
            },
            "ratings": 4.0,
            "reviews_count": 78,
            "warranty_months": 36,
            "added_date": "2024-11-05",
            "tags": [
                "Audio",
                "Samsung",
                "Mid-range"
            ]
        },
        {
            "id": 100007,
            "product_name": "Product-100007",
            "brand": "HyperX",
            "price": 312.5,
            "discount_percent": 11.0,
            "discounted_price": 278.12,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Networking",
            "sub_category": "Sub-Networking",
            "description": "High-quality Networking product with premium features",
            "specifications": {
                "weight": "1.1kg",
                "dimensions": "26x21x11cm",
                "color": "Black"
            },
            "ratings": 4.1,
            "reviews_count": 91,
            "warranty_months": 12,
            "added_date": "2024-10-31",
            "tags": [
                "Networking",
                "HyperX",
                "Premium"
            ]
        },
        {
            "id": 100008,
            "product_name": "Product-100008",
            "brand": "MSI",
            "price": 356.25,
            "discount_percent": 12.0,
            "discounted_price": 313.5,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Gaming Peripherals",
            "sub_category": "Sub-Gaming Peripherals",
            "description": "High-quality Gaming Peripherals product with premium features",
            "specifications": {
                "weight": "1.2kg",
                "dimensions": "27x22x12cm",
                "color": "White"
            },
            "ratings": 4.2,
            "reviews_count": 104,
            "warranty_months": 24,
            "added_date": "2024-10-26",
            "tags": [
                "Gaming Peripherals",
                "MSI",
                "Budget"
            ]
        },
        {
            "id": 100009,
            "product_name": "Product-100009",
            "brand": "Razer",
            "price": 400.0,
            "discount_percent": 13.0,
            "discounted_price": 348.0,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Monitors",
            "sub_category": "Sub-Monitors",
            "description": "High-quality Monitors product with premium features",
            "specifications": {
                "weight": "1.3kg",
                "dimensions": "28x23x13cm",
                "color": "Gray"
            },
            "ratings": 4.3,
            "reviews_count": 117,
            "warranty_months": 36,
            "added_date": "2024-10-21",
            "tags": [
                "Monitors",
                "Razer",
                "Mid-range"
            ]
        },
        {
            "id": 100010,
            "product_name": "Product-100010",
            "brand": "Logitech",
            "price": 443.75,
            "discount_percent": 14.0,
            "discounted_price": 381.62,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Laptops",
            "sub_category": "Sub-Laptops",
            "description": "High-quality Laptops product with premium features",
            "specifications": {
                "weight": "1.4kg",
                "dimensions": "29x24x14cm",
                "color": "Black"
            },
            "ratings": 4.4,
            "reviews_count": 130,
            "warranty_months": 12,
            "added_date": "2024-10-16",
            "tags": [
                "Laptops",
                "Logitech",
                "Premium"
            ]
        },
        {
            "id": 100011,
            "product_name": "Product-100011",
            "brand": "Corsair",
            "price": 487.5,
            "discount_percent": 15.0,
            "discounted_price": 414.38,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Components",
            "sub_category": "Sub-Components",
            "description": "High-quality Components product with premium features",
            "specifications": {
                "weight": "1.5kg",
                "dimensions": "30x25x15cm",
                "color": "White"
            },
            "ratings": 4.5,
            "reviews_count": 143,
            "warranty_months": 24,
            "added_date": "2024-10-11",
            "tags": [
                "Components",
                "Corsair",
                "Budget"
            ]
        },
        {
            "id": 100012,
            "product_name": "Product-100012",
            "brand": "ASUS",
            "price": 531.25,
            "discount_percent": 16.0,
            "discounted_price": 446.25,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Storage",
            "sub_category": "Sub-Storage",
            "description": "High-quality Storage product with premium features",
            "specifications": {
                "weight": "1.6kg",
                "dimensions": "31x26x16cm",
                "color": "Gray"
            },
            "ratings": 4.6,
            "reviews_count": 156,
            "warranty_months": 36,
            "added_date": "2024-10-06",
            "tags": [
                "Storage",
                "ASUS",
                "Mid-range"
            ]
        },
        {
            "id": 100013,
            "product_name": "Product-100013",
            "brand": "Dell",
            "price": 575.0,
            "discount_percent": 17.0,
            "discounted_price": 477.25,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Audio",
            "sub_category": "Sub-Audio",
            "description": "High-quality Audio product with premium features",
            "specifications": {
                "weight": "1.7kg",
                "dimensions": "32x27x17cm",
                "color": "Black"
            },
            "ratings": 4.7,
            "reviews_count": 169,
            "warranty_months": 12,
            "added_date": "2024-10-01",
            "tags": [
                "Audio",
                "Dell",
                "Premium"
            ]
        },
        {
            "id": 100014,
            "product_name": "Product-100014",
            "brand": "Samsung",
            "price": 618.75,
            "discount_percent": 18.0,
            "discounted_price": 507.38,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Networking",
            "sub_category": "Sub-Networking",
            "description": "High-quality Networking product with premium features",
            "specifications": {
                "weight": "1.8kg",
                "dimensions": "33x28x18cm",
                "color": "White"
            },
            "ratings": 4.8,
            "reviews_count": 182,
            "warranty_months": 24,
            "added_date": "2024-09-26",
            "tags": [
                "Networking",
                "Samsung",
                "Budget"
            ]
        },
        {
            "id": 100015,
            "product_name": "Product-100015",
            "brand": "HyperX",
            "price": 662.5,
            "discount_percent": 19.0,
            "discounted_price": 536.62,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Gaming Peripherals",
            "sub_category": "Sub-Gaming Peripherals",
            "description": "High-quality Gaming Peripherals product with premium features",
            "specifications": {
                "weight": "1.9kg",
                "dimensions": "34x29x19cm",
                "color": "Gray"
            },
            "ratings": 4.9,
            "reviews_count": 195,
            "warranty_months": 36,
            "added_date": "2024-09-21",
            "tags": [
                "Gaming Peripherals",
                "HyperX",
                "Mid-range"
            ]
        },
        {
            "id": 100016,
            "product_name": "Product-100016",
            "brand": "MSI",
            "price": 706.25,
            "discount_percent": 20.0,
            "discounted_price": 565.0,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Monitors",
            "sub_category": "Sub-Monitors",
            "description": "High-quality Monitors product with premium features",
            "specifications": {
                "weight": "2.0kg",
                "dimensions": "35x30x20cm",
                "color": "Black"
            },
            "ratings": 3.5,
            "reviews_count": 208,
            "warranty_months": 12,
            "added_date": "2024-09-16",
            "tags": [
                "Monitors",
                "MSI",
                "Premium"
            ]
        },
        {
            "id": 100017,
            "product_name": "Product-100017",
            "brand": "Razer",
            "price": 750.0,
            "discount_percent": 21.0,
            "discounted_price": 592.5,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Laptops",
            "sub_category": "Sub-Laptops",
            "description": "High-quality Laptops product with premium features",
            "specifications": {
                "weight": "2.1kg",
                "dimensions": "36x31x21cm",
                "color": "White"
            },
            "ratings": 3.6,
            "reviews_count": 221,
            "warranty_months": 24,
            "added_date": "2024-09-11",
            "tags": [
                "Laptops",
                "Razer",
                "Budget"
            ]
        },
        {
            "id": 100018,
            "product_name": "Product-100018",
            "brand": "Logitech",
            "price": 793.75,
            "discount_percent": 22.0,
            "discounted_price": 619.12,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Components",
            "sub_category": "Sub-Components",
            "description": "High-quality Components product with premium features",
            "specifications": {
                "weight": "2.2kg",
                "dimensions": "37x32x22cm",
                "color": "Gray"
            },
            "ratings": 3.7,
            "reviews_count": 234,
            "warranty_months": 36,
            "added_date": "2024-09-06",
            "tags": [
                "Components",
                "Logitech",
                "Mid-range"
            ]
        },
        {
            "id": 100019,
            "product_name": "Product-100019",
            "brand": "Corsair",
            "price": 837.5,
            "discount_percent": 23.0,
            "discounted_price": 644.88,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Storage",
            "sub_category": "Sub-Storage",
            "description": "High-quality Storage product with premium features",
            "specifications": {
                "weight": "2.3kg",
                "dimensions": "38x33x23cm",
                "color": "Black"
            },
            "ratings": 3.8,
            "reviews_count": 247,
            "warranty_months": 12,
            "added_date": "2024-09-01",
            "tags": [
                "Storage",
                "Corsair",
                "Premium"
            ]
        },
        {
            "id": 100020,
            "product_name": "Product-100020",
            "brand": "ASUS",
            "price": 881.25,
            "discount_percent": 24.0,
            "discounted_price": 669.75,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Audio",
            "sub_category": "Sub-Audio",
            "description": "High-quality Audio product with premium features",
            "specifications": {
                "weight": "2.4kg",
                "dimensions": "39x34x24cm",
                "color": "White"
            },
            "ratings": 3.9,
            "reviews_count": 260,
            "warranty_months": 24,
            "added_date": "2024-08-27",
            "tags": [
                "Audio",
                "ASUS",
                "Budget"
            ]
        },
        {
            "id": 100021,
            "product_name": "Product-100021",
            "brand": "Dell",
            "price": 925.0,
            "discount_percent": 5.0,
            "discounted_price": 878.75,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Networking",
            "sub_category": "Sub-Networking",
            "description": "High-quality Networking product with premium features",
            "specifications": {
                "weight": "2.5kg",
                "dimensions": "40x35x25cm",
                "color": "Gray"
            },
            "ratings": 4.0,
            "reviews_count": 273,
            "warranty_months": 36,
            "added_date": "2024-08-22",
            "tags": [
                "Networking",
                "Dell",
                "Mid-range"
            ]
        },
        {
            "id": 100022,
            "product_name": "Product-100022",
            "brand": "Samsung",
            "price": 968.75,
            "discount_percent": 6.0,
            "discounted_price": 910.62,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Gaming Peripherals",
            "sub_category": "Sub-Gaming Peripherals",
            "description": "High-quality Gaming Peripherals product with premium features",
            "specifications": {
                "weight": "2.6kg",
                "dimensions": "41x36x26cm",
                "color": "Black"
            },
            "ratings": 4.1,
            "reviews_count": 286,
            "warranty_months": 12,
            "added_date": "2024-08-17",
            "tags": [
                "Gaming Peripherals",
                "Samsung",
                "Premium"
            ]
        },
        {
            "id": 100023,
            "product_name": "Product-100023",
            "brand": "HyperX",
            "price": 1012.5,
            "discount_percent": 7.0,
            "discounted_price": 941.62,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Monitors",
            "sub_category": "Sub-Monitors",
            "description": "High-quality Monitors product with premium features",
            "specifications": {
                "weight": "2.7kg",
                "dimensions": "42x37x27cm",
                "color": "White"
            },
            "ratings": 4.2,
            "reviews_count": 299,
            "warranty_months": 24,
            "added_date": "2024-08-12",
            "tags": [
                "Monitors",
                "HyperX",
                "Budget"
            ]
        },
        {
            "id": 100024,
            "product_name": "Product-100024",
            "brand": "MSI",
            "price": 1056.25,
            "discount_percent": 8.0,
            "discounted_price": 971.75,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Laptops",
            "sub_category": "Sub-Laptops",
            "description": "High-quality Laptops product with premium features",
            "specifications": {
                "weight": "2.8kg",
                "dimensions": "43x38x28cm",
                "color": "Gray"
            },
            "ratings": 4.3,
            "reviews_count": 312,
            "warranty_months": 36,
            "added_date": "2024-08-07",
            "tags": [
                "Laptops",
                "MSI",
                "Mid-range"
            ]
        },
        {
            "id": 100025,
            "product_name": "Product-100025",
            "brand": "Razer",
            "price": 1100.0,
            "discount_percent": 9.0,
            "discounted_price": 1001.0,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Components",
            "sub_category": "Sub-Components",
            "description": "High-quality Components product with premium features",
            "specifications": {
                "weight": "2.9kg",
                "dimensions": "44x39x29cm",
                "color": "Black"
            },
            "ratings": 4.4,
            "reviews_count": 325,
            "warranty_months": 12,
            "added_date": "2024-08-02",
            "tags": [
                "Components",
                "Razer",
                "Premium"
            ]
        },
        {
            "id": 100026,
            "product_name": "Product-100026",
            "brand": "Logitech",
            "price": 1143.75,
            "discount_percent": 10.0,
            "discounted_price": 1029.38,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Storage",
            "sub_category": "Sub-Storage",
            "description": "High-quality Storage product with premium features",
            "specifications": {
                "weight": "3.0kg",
                "dimensions": "45x40x30cm",
                "color": "White"
            },
            "ratings": 4.5,
            "reviews_count": 338,
            "warranty_months": 24,
            "added_date": "2024-07-28",
            "tags": [
                "Storage",
                "Logitech",
                "Budget"
            ]
        },
        {
            "id": 100027,
            "product_name": "Product-100027",
            "brand": "Corsair",
            "price": 1187.5,
            "discount_percent": 11.0,
            "discounted_price": 1056.88,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Audio",
            "sub_category": "Sub-Audio",
            "description": "High-quality Audio product with premium features",
            "specifications": {
                "weight": "3.1kg",
                "dimensions": "46x41x31cm",
                "color": "Gray"
            },
            "ratings": 4.6,
            "reviews_count": 351,
            "warranty_months": 36,
            "added_date": "2024-07-23",
            "tags": [
                "Audio",
                "Corsair",
                "Mid-range"
            ]
        },
        {
            "id": 100028,
            "product_name": "Product-100028",
            "brand": "ASUS",
            "price": 1231.25,
            "discount_percent": 12.0,
            "discounted_price": 1083.5,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Networking",
            "sub_category": "Sub-Networking",
            "description": "High-quality Networking product with premium features",
            "specifications": {
                "weight": "3.2kg",
                "dimensions": "47x42x32cm",
                "color": "Black"
            },
            "ratings": 4.7,
            "reviews_count": 364,
            "warranty_months": 12,
            "added_date": "2024-07-18",
            "tags": [
                "Networking",
                "ASUS",
                "Premium"
            ]
        },
        {
            "id": 100029,
            "product_name": "Product-100029",
            "brand": "Dell",
            "price": 1275.0,
            "discount_percent": 13.0,
            "discounted_price": 1109.25,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Gaming Peripherals",
            "sub_category": "Sub-Gaming Peripherals",
            "description": "High-quality Gaming Peripherals product with premium features",
            "specifications": {
                "weight": "3.3kg",
                "dimensions": "48x43x33cm",
                "color": "White"
            },
            "ratings": 4.8,
            "reviews_count": 377,
            "warranty_months": 24,
            "added_date": "2024-07-13",
            "tags": [
                "Gaming Peripherals",
                "Dell",
                "Budget"
            ]
        },
        {
            "id": 100030,
            "product_name": "Product-100030",
            "brand": "Samsung",
            "price": 1318.75,
            "discount_percent": 14.0,
            "discounted_price": 1134.12,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Monitors",
            "sub_category": "Sub-Monitors",
            "description": "High-quality Monitors product with premium features",
            "specifications": {
                "weight": "3.4kg",
                "dimensions": "49x44x34cm",
                "color": "Gray"
            },
            "ratings": 4.9,
            "reviews_count": 390,
            "warranty_months": 36,
            "added_date": "2024-07-08",
            "tags": [
                "Monitors",
                "Samsung",
                "Mid-range"
            ]
        },
        {
            "id": 100031,
            "product_name": "Product-100031",
            "brand": "HyperX",
            "price": 1362.5,
            "discount_percent": 15.0,
            "discounted_price": 1158.12,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Laptops",
            "sub_category": "Sub-Laptops",
            "description": "High-quality Laptops product with premium features",
            "specifications": {
                "weight": "3.5kg",
                "dimensions": "50x45x35cm",
                "color": "Black"
            },
            "ratings": 3.5,
            "reviews_count": 403,
            "warranty_months": 12,
            "added_date": "2024-07-03",
            "tags": [
                "Laptops",
                "HyperX",
                "Premium"
            ]
        },
        {
            "id": 100032,
            "product_name": "Product-100032",
            "brand": "MSI",
            "price": 1406.25,
            "discount_percent": 16.0,
            "discounted_price": 1181.25,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Components",
            "sub_category": "Sub-Components",
            "description": "High-quality Components product with premium features",
            "specifications": {
                "weight": "3.6kg",
                "dimensions": "51x46x36cm",
                "color": "White"
            },
            "ratings": 3.6,
            "reviews_count": 416,
            "warranty_months": 24,
            "added_date": "2024-06-28",
            "tags": [
                "Components",
                "MSI",
                "Budget"
            ]
        },
        {
            "id": 100033,
            "product_name": "Product-100033",
            "brand": "Razer",
            "price": 1450.0,
            "discount_percent": 17.0,
            "discounted_price": 1203.5,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Storage",
            "sub_category": "Sub-Storage",
            "description": "High-quality Storage product with premium features",
            "specifications": {
                "weight": "3.7kg",
                "dimensions": "52x47x37cm",
                "color": "Gray"
            },
            "ratings": 3.7,
            "reviews_count": 429,
            "warranty_months": 36,
            "added_date": "2024-06-23",
            "tags": [
                "Storage",
                "Razer",
                "Mid-range"
            ]
        },
        {
            "id": 100034,
            "product_name": "Product-100034",
            "brand": "Logitech",
            "price": 1493.75,
            "discount_percent": 18.0,
            "discounted_price": 1224.88,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Audio",
            "sub_category": "Sub-Audio",
            "description": "High-quality Audio product with premium features",
            "specifications": {
                "weight": "3.8kg",
                "dimensions": "53x48x38cm",
                "color": "Black"
            },
            "ratings": 3.8,
            "reviews_count": 442,
            "warranty_months": 12,
            "added_date": "2024-06-18",
            "tags": [
                "Audio",
                "Logitech",
                "Premium"
            ]
        },
        {
            "id": 100035,
            "product_name": "Product-100035",
            "brand": "Corsair",
            "price": 1537.5,
            "discount_percent": 19.0,
            "discounted_price": 1245.38,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Networking",
            "sub_category": "Sub-Networking",
            "description": "High-quality Networking product with premium features",
            "specifications": {
                "weight": "3.9kg",
                "dimensions": "54x49x39cm",
                "color": "White"
            },
            "ratings": 3.9,
            "reviews_count": 455,
            "warranty_months": 24,
            "added_date": "2024-06-13",
            "tags": [
                "Networking",
                "Corsair",
                "Budget"
            ]
        },
        {
            "id": 100036,
            "product_name": "Product-100036",
            "brand": "ASUS",
            "price": 1581.25,
            "discount_percent": 20.0,
            "discounted_price": 1265.0,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Gaming Peripherals",
            "sub_category": "Sub-Gaming Peripherals",
            "description": "High-quality Gaming Peripherals product with premium features",
            "specifications": {
                "weight": "4.0kg",
                "dimensions": "55x50x40cm",
                "color": "Gray"
            },
            "ratings": 4.0,
            "reviews_count": 468,
            "warranty_months": 36,
            "added_date": "2024-06-08",
            "tags": [
                "Gaming Peripherals",
                "ASUS",
                "Mid-range"
            ]
        },
        {
            "id": 100037,
            "product_name": "Product-100037",
            "brand": "Dell",
            "price": 1625.0,
            "discount_percent": 21.0,
            "discounted_price": 1283.75,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Monitors",
            "sub_category": "Sub-Monitors",
            "description": "High-quality Monitors product with premium features",
            "specifications": {
                "weight": "4.1kg",
                "dimensions": "56x51x41cm",
                "color": "Black"
            },
            "ratings": 4.1,
            "reviews_count": 481,
            "warranty_months": 12,
            "added_date": "2024-06-03",
            "tags": [
                "Monitors",
                "Dell",
                "Premium"
            ]
        },
        {
            "id": 100038,
            "product_name": "Product-100038",
            "brand": "Samsung",
            "price": 1668.75,
            "discount_percent": 22.0,
            "discounted_price": 1301.62,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Laptops",
            "sub_category": "Sub-Laptops",
            "description": "High-quality Laptops product with premium features",
            "specifications": {
                "weight": "4.2kg",
                "dimensions": "57x52x42cm",
                "color": "White"
            },
            "ratings": 4.2,
            "reviews_count": 494,
            "warranty_months": 24,
            "added_date": "2024-05-29",
            "tags": [
                "Laptops",
                "Samsung",
                "Budget"
            ]
        },
        {
            "id": 100039,
            "product_name": "Product-100039",
            "brand": "HyperX",
            "price": 1712.5,
            "discount_percent": 23.0,
            "discounted_price": 1318.62,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Components",
            "sub_category": "Sub-Components",
            "description": "High-quality Components product with premium features",
            "specifications": {
                "weight": "4.3kg",
                "dimensions": "58x53x43cm",
                "color": "Gray"
            },
            "ratings": 4.3,
            "reviews_count": 507,
            "warranty_months": 36,
            "added_date": "2024-05-24",
            "tags": [
                "Components",
                "HyperX",
                "Mid-range"
            ]
        },
        {
            "id": 100040,
            "product_name": "Product-100040",
            "brand": "MSI",
            "price": 1756.25,
            "discount_percent": 24.0,
            "discounted_price": 1334.75,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Storage",
            "sub_category": "Sub-Storage",
            "description": "High-quality Storage product with premium features",
            "specifications": {
                "weight": "4.4kg",
                "dimensions": "59x54x44cm",
                "color": "Black"
            },
            "ratings": 4.4,
            "reviews_count": 520,
            "warranty_months": 12,
            "added_date": "2024-05-19",
            "tags": [
                "Storage",
                "MSI",
                "Premium"
            ]
        },
        {
            "id": 100041,
            "product_name": "Product-100041",
            "brand": "Razer",
            "price": 1800.0,
            "discount_percent": 5.0,
            "discounted_price": 1710.0,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Audio",
            "sub_category": "Sub-Audio",
            "description": "High-quality Audio product with premium features",
            "specifications": {
                "weight": "4.5kg",
                "dimensions": "60x55x45cm",
                "color": "White"
            },
            "ratings": 4.5,
            "reviews_count": 533,
            "warranty_months": 24,
            "added_date": "2024-05-14",
            "tags": [
                "Audio",
                "Razer",
                "Budget"
            ]
        },
        {
            "id": 100042,
            "product_name": "Product-100042",
            "brand": "Logitech",
            "price": 1843.75,
            "discount_percent": 6.0,
            "discounted_price": 1733.12,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Networking",
            "sub_category": "Sub-Networking",
            "description": "High-quality Networking product with premium features",
            "specifications": {
                "weight": "4.6kg",
                "dimensions": "61x56x46cm",
                "color": "Gray"
            },
            "ratings": 4.6,
            "reviews_count": 546,
            "warranty_months": 36,
            "added_date": "2024-05-09",
            "tags": [
                "Networking",
                "Logitech",
                "Mid-range"
            ]
        },
        {
            "id": 100043,
            "product_name": "Product-100043",
            "brand": "Corsair",
            "price": 1887.5,
            "discount_percent": 7.0,
            "discounted_price": 1755.37,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Gaming Peripherals",
            "sub_category": "Sub-Gaming Peripherals",
            "description": "High-quality Gaming Peripherals product with premium features",
            "specifications": {
                "weight": "4.7kg",
                "dimensions": "62x57x47cm",
                "color": "Black"
            },
            "ratings": 4.7,
            "reviews_count": 559,
            "warranty_months": 12,
            "added_date": "2024-05-04",
            "tags": [
                "Gaming Peripherals",
                "Corsair",
                "Premium"
            ]
        },
        {
            "id": 100044,
            "product_name": "Product-100044",
            "brand": "ASUS",
            "price": 1931.25,
            "discount_percent": 8.0,
            "discounted_price": 1776.75,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Monitors",
            "sub_category": "Sub-Monitors",
            "description": "High-quality Monitors product with premium features",
            "specifications": {
                "weight": "4.8kg",
                "dimensions": "63x58x48cm",
                "color": "White"
            },
            "ratings": 4.8,
            "reviews_count": 572,
            "warranty_months": 24,
            "added_date": "2024-04-29",
            "tags": [
                "Monitors",
                "ASUS",
                "Budget"
            ]
        },
        {
            "id": 100045,
            "product_name": "Product-100045",
            "brand": "Dell",
            "price": 1975.0,
            "discount_percent": 9.0,
            "discounted_price": 1797.25,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Laptops",
            "sub_category": "Sub-Laptops",
            "description": "High-quality Laptops product with premium features",
            "specifications": {
                "weight": "4.9kg",
                "dimensions": "64x59x49cm",
                "color": "Gray"
            },
            "ratings": 4.9,
            "reviews_count": 585,
            "warranty_months": 36,
            "added_date": "2024-04-24",
            "tags": [
                "Laptops",
                "Dell",
                "Mid-range"
            ]
        },
        {
            "id": 100046,
            "product_name": "Product-100046",
            "brand": "Samsung",
            "price": 2018.75,
            "discount_percent": 10.0,
            "discounted_price": 1816.88,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Components",
            "sub_category": "Sub-Components",
            "description": "High-quality Components product with premium features",
            "specifications": {
                "weight": "5.0kg",
                "dimensions": "65x60x50cm",
                "color": "Black"
            },
            "ratings": 3.5,
            "reviews_count": 598,
            "warranty_months": 12,
            "added_date": "2024-04-19",
            "tags": [
                "Components",
                "Samsung",
                "Premium"
            ]
        },
        {
            "id": 100047,
            "product_name": "Product-100047",
            "brand": "HyperX",
            "price": 2062.5,
            "discount_percent": 11.0,
            "discounted_price": 1835.62,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Storage",
            "sub_category": "Sub-Storage",
            "description": "High-quality Storage product with premium features",
            "specifications": {
                "weight": "5.1kg",
                "dimensions": "66x61x51cm",
                "color": "White"
            },
            "ratings": 3.6,
            "reviews_count": 611,
            "warranty_months": 24,
            "added_date": "2024-04-14",
            "tags": [
                "Storage",
                "HyperX",
                "Budget"
            ]
        },
        {
            "id": 100048,
            "product_name": "Product-100048",
            "brand": "MSI",
            "price": 2106.25,
            "discount_percent": 12.0,
            "discounted_price": 1853.5,
            "in_stock": True,
            "stock_quantity": 50,
            "category": "Audio",
            "sub_category": "Sub-Audio",
            "description": "High-quality Audio product with premium features",
            "specifications": {
                "weight": "5.2kg",
                "dimensions": "67x62x52cm",
                "color": "Gray"
            },
            "ratings": 3.7,
            "reviews_count": 624,
            "warranty_months": 36,
            "added_date": "2024-04-09",
            "tags": [
                "Audio",
                "MSI",
                "Mid-range"
            ]
        },
        {
            "id": 100049,
            "product_name": "Product-100049",
            "brand": "Razer",
            "price": 2150.0,
            "discount_percent": 13.0,
            "discounted_price": 1870.5,
            "in_stock": False,
            "stock_quantity": 0,
            "category": "Networking",
            "sub_category": "Sub-Networking",
            "description": "High-quality Networking product with premium features",
            "specifications": {
                "weight": "5.3kg",
                "dimensions": "68x63x53cm",
                "color": "Black"
            },
            "ratings": 3.8,
            "reviews_count": 637,
            "warranty_months": 12,
            "added_date": "2024-04-04",
            "tags": [
                "Networking",
                "Razer",
                "Premium"
            ]
        },
        {
            "id": 100050,
            "product_name": "Product-100050",
            "brand": "Logitech",
            "price": 2193.75,
            "discount_percent": 14.0,
            "discounted_price": 1886.62,
            "in_stock": True,
            "stock_quantity": 25,
            "category": "Gaming Peripherals",
            "sub_category": "Sub-Gaming Peripherals",
            "description": "High-quality Gaming Peripherals product with premium features",
            "specifications": {
                "weight": "5.4kg",
                "dimensions": "69x64x54cm",
                "color": "White"
            },
            "ratings": 3.9,
            "reviews_count": 650,
            "warranty_months": 24,
            "added_date": "2024-03-30",
            "tags": [
                "Gaming Peripherals",
                "Logitech",
                "Budget"
            ]
        }
    ]

