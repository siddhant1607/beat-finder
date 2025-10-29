from pymongo import MongoClient
import os

def get_db():
    try:
        import streamlit as st
        mongo_url = st.secrets["MONGO_URL"]
    except Exception:
        mongo_url = os.environ.get("MONGO_URL")
        if not mongo_url:
            raise ValueError("Please set MONGO_URL as an environment variable or in Streamlit secrets.")
    client = MongoClient(mongo_url)
    db = client['fingerprint_db']
    return db['hashes']
