import streamlit as st
import pandas as pd
import re
from difflib import get_close_matches
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------
# Config
# -------------------------------
DATA_PATH = "ABC.xlsx"
st.set_page_config(page_title="📦 Edmond's Inventory Chatbot (Excel + RAG)", layout="wide")

# -------------------------------
# Load Excel + Build RAG Pipeline
# -------------------------------
@st.cache_resource
def load_rag_pipeline():
    all_sheets = pd.read_excel(DATA_PATH, sheet_name=None, header=1)
    dfs = []
    for name, df in all_sheets.items():
        df = df.dropna(how="all").ffill().fillna("N/A")
        df["Sheet"] = name
        dfs.append(df)
    df_full = pd.concat(dfs, ignore_index=True)

    # Prepare documents
    docs = []
    for _, row in df_full.iterrows():
        content = (
            f"Sheet: {row['Sheet']}, Case: {row['Case']}, "
            f"Item: {row['Item']}, Units: {row['Units']}, Remarks: {row['Remarks']}"
        )
        docs.append(Document(page_content=content, metadata=row.to_dict()))

    # FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Load Flan-T5
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return df_full, qa


df, qa = load_rag_pipeline()

# -------------------------------
# Helper functions (same as app.py)
# -------------------------------
def normalize_query(q):
    q = q.lower()
    replacements = {
        # Units synonyms
        "quantity": "units",
        "quantities": "units",
        "count": "units",
        "amount": "units",
        "how many": "units",

        # Remarks/usage synonyms
        "usage": "remarks",
        "used for": "remarks",
        "use for": "remarks",
        "purpose": "remarks",
        "note": "remarks",
        "notes": "remarks",
        "comment": "remarks",
        "comments": "remarks",
        "detail": "remarks",
        "details": "remarks",
        "explanation": "remarks",
        "function": "remarks",
    }
    for k, v in replacements.items():
        q = q.replace(k, v)
    return q


def extract_after(keywords, q):
    for k in keywords:
        if k in q:
            return q.split(k, 1)[-1].strip(" :?!.;'\"")
    return q

def fuzzy_filter(df, item, topn=6, cutoff=0.4):
    item_norm = re.sub(r"[^a-z0-9]+", " ", item.lower())
    items_norm = df["Item"].astype(str).apply(lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()))
    cand = get_close_matches(item_norm, items_norm.tolist(), n=topn, cutoff=cutoff)
    mask = items_norm.isin(cand) | df["Item"].str.lower().str.contains(item.lower(), na=False)
    return df[mask]

def structured_lookup(query, df):
    q = normalize_query(query)

    # --- Detect case or sheet in query ---
    case_match = re.search(r"\b([A-Za-z]\d+)\b", q)
    sheet_match = re.search(r"sheet\s*([A-Za-z])", q, re.IGNORECASE)

    # Start with full dataset
    subset = df.copy()
    case_name, sheet_name = None, None

    if sheet_match:
        sheet_name = sheet_match.group(1).upper()
        subset = subset[subset["Sheet"].astype(str).str.upper() == sheet_name]

    if case_match:
        case_name = case_match.group(1).upper()
        subset = subset[subset["Case"].astype(str).str.upper() == case_name]

    # Helper: fallback if subset is empty
    def smart_search(item_phrase):
        matches = fuzzy_filter(subset, item_phrase)
        if matches.empty:  # fallback to full search if no match in subset
            matches = fuzzy_filter(df, item_phrase)
        return matches

    # --- Handle different query types ---
    if any(p in q for p in ["where is", "which case", "location of"]):
        item_phrase = extract_after(["where is", "which case", "location of"], q)
        matches = smart_search(item_phrase)
        if matches.empty:
            return {"answer": f"Could not locate '{item_phrase}'."}
        summary = ", ".join(matches["Case"].astype(str).unique())
        out = [f"{row['Item']} ➜ Case {row['Case']} (Sheet {row['Sheet']})" for _, row in matches.iterrows()]
        return {"answer": [f"Found in cases: {summary}"] + out}

    # Units
    if "units" in q:
        item_phrase = extract_after(["units for", "units of", "units"], q)
        matches = smart_search(item_phrase)
        if not matches.empty:
            out = [f"{row['Item']} ➜ Units: {row['Units']} (Case {row['Case']}, Sheet {row['Sheet']})"
                   for _, row in matches.head(5).iterrows()]
            return {"answer": out}

    # Remarks / usage / what is used for
    if any(p in q for p in ["remarks", "usage", "used for", "purpose", "what is"]):
        item_phrase = extract_after(["remarks for", "usage for", "used for", "purpose of", "what is"], q)
        matches = smart_search(item_phrase)
        if not matches.empty:
            out = [f"{row['Item']} ➜ {row['Remarks']} (Case {row['Case']}, Sheet {row['Sheet']})"
                   for _, row in matches.head(5).iterrows()]
            return {"answer": out}

    return None



def ask_question_local(query):
    structured = structured_lookup(query, df)
    if structured:
        return structured
    result = qa(query)
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:150] for doc in result["source_documents"]],
    }

# -------------------------------
# Sidebar for browsing
# -------------------------------
st.sidebar.header("📦 Browse Inventory")
sheet = st.sidebar.selectbox("📄 Select Sheet", sorted(df["Sheet"].unique()))
df_sheet = df[df["Sheet"] == sheet]
cases = sorted(df_sheet["Case"].astype(str).unique())
case = st.sidebar.selectbox("📦 Select Case", cases)
df_case = df_sheet[df_sheet["Case"].astype(str) == case]
items = sorted(df_case["Item"].astype(str).unique())
item = st.sidebar.selectbox("🎯 Select Item", items)
selected_row = df_case[df_case["Item"] == item]

st.sidebar.markdown("—")
st.sidebar.dataframe(df_case[["Item", "Units", "Remarks"]], use_container_width=True, hide_index=True)

if not selected_row.empty:
    st.sidebar.markdown(f"**Units:** {selected_row.iloc[0]['Units']}")
    st.sidebar.markdown(f"**Remarks:** {selected_row.iloc[0]['Remarks']}")

# -------------------------------
# Main interface
# -------------------------------
st.title("📦 Edmond's Inventory Chatbot (Excel + RAG)")
st.caption("Ask naturally or explore using the buttons below.")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📍 Where is this item?"):
        resp = ask_question_local(f"where is {item}")
        st.write(resp["answer"])
with col2:
    if st.button("🔢 Units"):
        resp = ask_question_local(f"units for {item}")
        st.write(resp["answer"])
with col3:
    if st.button("📝 Usage / Remarks"):
        resp = ask_question_local(f"what is {item} used for")
        st.write(resp["answer"])

st.markdown("---")

st.markdown("---")

# -------------------------------
# 💬 Free-form Chat + Example Questions
# -------------------------------
st.subheader("💬 Ask your own question")

# --- Example questions ---
examples = [
    f"where is {item}",
    f"units for {item}",
    f"what is {item} used for",
    f"remarks for {item}",
    f"items in case {case}",
    "find spotlight",
    "which items have less than 3 units in sheet C",
    "summarize case A1"
]

st.caption("💡 Click an example below to auto-fill the question box:")

cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(ex):
        st.session_state["q"] = ex

# --- Input box for custom queries ---
q = st.text_input("Enter your question:", key="q")

if st.button("Ask", type="primary"):
    resp = ask_question_local(q)
    st.subheader("🧾 Answer")

    ans = resp.get("answer", "")

    if isinstance(ans, list):
        for a in ans:
            # Try to parse pattern like "Item ➜ Units: X (Case Y, Sheet Z)"
            if "➜" in a and "Units:" in a and "(" in a:
                try:
                    item_part = a.split("➜")[0].strip()
                    units_part = a.split("Units:")[1].split("(")[0].strip()

                    # Extract Case and Sheet info (if available)
                    loc_text = a.split("(")[1].strip(")")
                    case_match = re.search(r"Case\s*([A-Za-z0-9]+)", loc_text)
                    sheet_match = re.search(r"Sheet\s*([A-Za-z])", loc_text)
                    case_text = case_match.group(1) if case_match else "N/A"
                    sheet_text = sheet_match.group(1) if sheet_match else "N/A"

                    # Clean formatted display
                    st.markdown(f"""
                    **🧾 Item:** {item_part}  
                    **📦 Units:** {units_part}  
                    **📍 Case:** {case_text}  
                    **📄 Sheet:** {sheet_text}
                    """)
                    st.divider()
                except Exception:
                    st.write(a)
            else:
                st.write(a)
    else:
        st.write(ans)

    # Optional: show retrieved context for transparency
    if "sources" in resp and resp["sources"]:
        with st.expander("📚 Retrieved context (from Excel rows)"):
            for s in resp["sources"]:
                st.write(f"- {s}")

else:
    st.caption("💬 Tip: Use the example buttons above for reliable demo queries.")

st.markdown("---")
st.caption("🧠 Powered by LangChain + FAISS + Flan-T5 (Local Excel RAG Chatbot).")

