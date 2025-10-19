import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import Tool
from langchain_community.cache import InMemoryCache
from prompts import rag_prompt, rag_tool_prompt
from config import config

# -------------------------------------------------------------------
# 1. CACHE
# -------------------------------------------------------------------
from langchain.globals import set_llm_cache
set_llm_cache(InMemoryCache())

# -------------------------------------------------------------------
# 2. HELPER: EMBEDDING MODEL
# -------------------------------------------------------------------
def get_embedding_model():
    """Load or reuse HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=config["embedding"]["model_name"])

# -------------------------------------------------------------------
# 3. HELPER: LOAD DOCUMENTS & BUILD VECTOR STORE (once)
# -------------------------------------------------------------------
def load_vectorstore():
    """Load (or build) Chroma vectorstore."""
    doc_dir = Path(config["document"]["doc_dir"])
    chroma_path = "chroma_db"

    # Reuse existing Chroma DB if already built
    if os.path.exists(chroma_path) and os.listdir(chroma_path):
        return Chroma(persist_directory=chroma_path, embedding_function=get_embedding_model())

    # Otherwise, rebuild from PDFs
    all_docs = []
    for pdf_file in doc_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = pdf_file.name
        all_docs.extend(docs)

    if not all_docs:
        raise FileNotFoundError(f"No PDFs found in {doc_dir}. Please add backpacker guides.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["document"]["chunk_size"],
        chunk_overlap=config["document"]["chunk_overlap"],
    )
    splits = text_splitter.split_documents(all_docs)

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=chroma_path,
    )

    return vectorstore

# -------------------------------------------------------------------
# 4. RETRIEVER + LLM
# -------------------------------------------------------------------
def get_retriever():
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 12},
    )

def get_llm():
    """Google Gemini (Generative AI) model."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=config["llm"]["model_name"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
        timeout=config["llm"]["timeout"],
        max_retries=config["llm"]["max_retries"],
    )

# -------------------------------------------------------------------
# 5. RAG CHAIN SETUP
# -------------------------------------------------------------------
retriever = get_retriever()
llm = get_llm()
combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# -------------------------------------------------------------------
# 6. SOURCE FORMATTING
# -------------------------------------------------------------------
def format_sources(docs):
    """Return clean, de-duplicated document citations."""
    seen, formatted = set(), []
    for d in docs:
        src = d.metadata.get("source", "Guide")
        page = d.metadata.get("page", None)
        label = f"{src} (p.{page + 1})" if isinstance(page, int) else src
        if label not in seen:
            formatted.append(f"- {label}")
            seen.add(label)
    return "\n".join(formatted) if formatted else "No sources found."

# -------------------------------------------------------------------
# 7. MAIN EXECUTION FUNCTION
# -------------------------------------------------------------------
def rag_with_sources(query: str):
    """Run RAG chain and return structured answer with sources."""
    result = retrieval_chain.invoke({"input": query})
    answer = result.get("answer", "No answer generated.")
    context_docs = result.get("context", [])
    # sources = format_sources(context_docs)
    return f"**Answer:** {answer}\n\n"

# -------------------------------------------------------------------
# 8. TOOL REGISTRATION
# -------------------------------------------------------------------
rag_tool = Tool(
    name="RAGTool",
    description=(
        "Use ONLY for questions answerable from the backpacker PDF guides. "
        "Ideal for travel routes, campsites, fees, safety tips, or attractions. "
        "Always cite document sources."
    ),
    func=rag_with_sources,
)

def rag_query(question: str) -> str:
    """Lightweight factual query for internal tools like budget planner."""
    try:
        combine_docs_chain = create_stuff_documents_chain(llm, rag_tool_prompt)
        temp_chain = create_retrieval_chain(retriever, combine_docs_chain)
        result = temp_chain.invoke({"input": question})
        return result.get("answer", "No info found in guide.")
    except Exception as e:
        return f"⚠️ RAG query error: {e}"