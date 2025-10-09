# DEMO 

### ⚠️The “Complete Project” – including FastAPI backend, LangChain RAG logic & Excel vectorization pipeline – is KEPT PRIVATE to prevent unauthorized copying on the PUBLIC INTERNET.

### 🤝 Full access can be HAPPILY shared ‘Upon Recruiter Request’ to showcase my Full-Stack AI & LLM skills.

### 👉 Try the live demo here: https://edmondproject2-6ptimyd3dnbgwopjjgdmxd.streamlit.app/

📄 The sample Excel file [ABC.xlsx](./ABC.xlsx) is included to let recruiters explore the data schema and test item queries interactively. 

---

📖 Project Overview: 💬 Edmond’s Inventory Chatbot System

This project demonstrates how AI-powered retrieval-augmented generation (RAG) can turn traditional Excel inventory files into a conversational assistant.
Users can ask natural-language questions such as:

“What is the case number for the 20W spotlight?”

“Which sheet is the DXR speaker listed under?”

“What is this item used for?”

The system retrieves and interprets information directly from Excel sheets and provides precise, context-aware answers.

---

🔍 Key Approaches

LangChain RAG pipeline – Converts Excel rows into semantic documents for intelligent retrieval.

Hugging Face embeddings – Generates dense vector representations for accurate semantic search.

FAISS vector store – Enables fast and efficient retrieval of relevant inventory entries.

FastAPI backend – Manages user queries and connects the RAG logic to the front-end.

Streamlit UI – Provides an intuitive chat interface for querying inventory data.

---

⚙️ Tech Stack

🐍 Python

🧠 LangChain / Hugging Face Embeddings

🧩 FAISS Vector Database

⚡ FastAPI (Backend)

💬 Streamlit (Frontend)

🧾 Pandas (Excel Processing)

🔠 re / difflib (String Similarity Matching)

✨ Features

Load and merge multiple Excel sheets dynamically.

Intelligent semantic search and Q&A over structured tabular data.

Extracts Remarks automatically when users ask about an item’s usage or function.

Built-in fuzzy matching for item names (handles typos and partial matches).

Deployed as a Streamlit web app.

---

📌 Notes for Recruiters

The GitHub repository contains only the demo app code (Streamlit frontend + partial FastAPI logic).
The complete RAG pipeline, embeddings, and deployment setup are private but can be shared upon request.

This project demonstrates end-to-end AI & LLM engineering skills — from data ingestion → semantic retrieval → language understanding → deployment.
