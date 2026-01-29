# 🧠 NeuroLitRAG

**AI-Powered Neuroscience Literature Search & Synthesis**

Built with **Cohere Embed + Rerank + Command** for intelligent scientific literature search.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## ✨ Features

- 🔍 **Semantic Search** — Find papers by meaning, not keywords
- 🎯 **Cohere Rerank** — Precise relevance scoring
- 📝 **Cited Answers** — Every claim backed by sources
- 🧬 **Neuroscience Focus** — Domain-optimized

---

## 🏗️ Architecture

```
Query → Cohere Embed → Vector Search → Cohere Rerank → Cohere Command → Answer
```

---

## 🚀 Try It

**Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app)

---

## 🛠️ Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/neuro-lit-rag.git
cd neuro-lit-rag

# Install
pip install -r requirements.txt

# Add your API key
echo "COHERE_API_KEY=your_key_here" > .env

# Run
streamlit run app.py
```

Get a free Cohere API key at: https://dashboard.cohere.com/api-keys

---

## 📁 Project Structure

```
neuro-lit-rag/
├── app.py              ← Streamlit web app
├── requirements.txt
├── src/
│   ├── embeddings.py   ← Cohere Embed
│   ├── reranker.py     ← Cohere Rerank
│   ├── generator.py    ← Cohere Command
│   ├── vector_store.py ← ChromaDB
│   └── pipeline.py     ← RAG orchestration
└── .streamlit/
    └── config.toml     ← Streamlit theme
```

---

## 🎯 Built For

This project demonstrates proficiency with:
- Cohere's full API stack (Embed, Rerank, Command)
- RAG pipeline architecture
- Production deployment

---

*Built with [Cohere](https://cohere.com/) 🚀*
