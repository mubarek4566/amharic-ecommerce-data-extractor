# amharic-ecommerce-data-extractor
This Project aim to increasing decentralization of Telegram channels used for online selling, vendors and customers face challenges in discovering products and managing transactions.


# Amharic E-commerce Data Extractor for EthioMart

## 🛍️ Project Overview

**EthioMart** aims to become the leading aggregator of Telegram-based e-commerce activity in Ethiopia. With the increasing decentralization of Telegram channels used for online selling, vendors and customers face challenges in discovering products and managing transactions. 

This project builds an automated system to extract and structure business-critical entities such as product names, prices, locations, and contact information from Amharic text and images shared in these Telegram channels.

---

## 🎯 Key Objectives

- ✅ Ingest and preprocess text and image data from Telegram-based e-commerce channels.
- ✅ Fine-tune transformer-based models (e.g., XLM-Roberta, mBERT) for Amharic Named Entity Recognition (NER).
- ✅ Evaluate different NER models and select the best one based on F1-score, Precision, and Recall.
- ✅ Use SHAP and LIME to explain the model’s decisions and improve trust.
- ✅ Link predictions to EthioMart's central platform for real-time product discovery.

---

## 🧠 Entities to Extract

- `PRODUCT`: Names or types of products
- `PRICE`: Prices or monetary mentions
- `LOCATION`: Mentioned locations (cities, areas)
- `DELIVERY_FEE` (optional): e.g. “Free delivery”, “150 birr delivery fee”
- `CONTACT_INFO` (optional): e.g. Phone numbers, @Telegram usernames

---

## 📊 Sample Workflow