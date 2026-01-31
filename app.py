"""
NeuroLitRAG - Debug Version
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="NeuroLitRAG Debug", page_icon="🧠")

st.title("🧠 NeuroLitRAG - Debug Mode")

# DEBUG: Check secrets
st.subheader("1. Checking Secrets...")

try:
    all_secrets = dict(st.secrets)
    st.write(f"✅ Secrets found: {list(all_secrets.keys())}")
except Exception as e:
    st.write(f"❌ Error accessing secrets: {e}")
    all_secrets = {}

# DEBUG: Try to get the key
st.subheader("2. Looking for COHERE_API_KEY...")

if "COHERE_API_KEY" in all_secrets:
    key = all_secrets["COHERE_API_KEY"]
    st.write(f"✅ Found in secrets! Starts with: {key[:10]}...")
    os.environ["COHERE_API_KEY"] = key
else:
    st.write("❌ COHERE_API_KEY not found in secrets")

# DEBUG: Check env var
st.subheader("3. Environment Variable...")

env_key = os.getenv("COHERE_API_KEY")
if env_key:
    st.write(f"✅ Env var set! Starts with: {env_key[:10]}...")
else:
    st.write("❌ Env var not set")

# Show what to do
st.subheader("4. What You Need")
st.code('COHERE_API_KEY = "your_key_here"', language="toml")
st.write("☝️ This exact format should be in your Streamlit Cloud Secrets")
