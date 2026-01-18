from core.Agent import Agent
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import create_retriever_tool
import math
from pathlib import Path
from datetime import datetime

@tool
def ask_human(prompt: str) -> str:
    """Ask the notebook user and return their answer."""
    return input(f"{prompt}\nYour answer: ")

def agent_as_tool(agent: Agent, name: str):
    def _call_agent(prompt: str) -> str:
        return agent.query(prompt)

    return StructuredTool.from_function(
        func=_call_agent,
        name=name,
        description=agent.role,
    )

def kb_search(path: str, k=3):
    kb_dir = Path(path)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # load existing DB if present
    if (kb_dir / "chroma.sqlite3").exists():
        db = Chroma(persist_directory=str(kb_dir), embedding_function=embeddings)

    # Otherwise build
    else:
        docs = load_docs(path) 
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)
        chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

        db = Chroma.from_documents(chunks, embeddings, persist_directory=str(kb_dir))
        db.persist()

        # Create manifest
        write_kb_manifest(kb_dir, docs)

        # Clear folder
        clear_raw_docs(kb_dir)


    retriever = db.as_retriever(search_kwargs={"k": k})
    return create_retriever_tool(
        retriever,
        name="search",
        description=f"Search the knowledge base at {path}",
    )


@tool
def calculate(expr: str) -> str:
    """Evaluate a weighted score expression."""
    try:
        return str(eval(expr, {"__builtins__": {}}, math.__dict__))
    except Exception as e:
        return f"ERROR: {e}"

def load_docs(path: str):
    docs = []
    docs += DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader,
                            loader_kwargs={"encoding":"utf-8"}).load()
    docs += DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader,
                            loader_kwargs={"encoding":"utf-8"}).load()
    docs += PyPDFDirectoryLoader(path).load()

    if not docs:
        raise ValueError(f"No documents found in {path}")

    return docs

def write_kb_manifest(kb_dir: str | Path, docs):
    kb_dir = Path(kb_dir)
    lines = []
    lines.append(f"# KB Manifest")
    lines.append(f"- built_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- documents: {len(docs)}")
    lines.append("")
    lines.append("| source | type |")
    lines.append("|---|---|")

    seen = set()
    for d in docs:
        src = d.metadata.get("source", "")
        if not src or src in seen:
            continue
        seen.add(src)
        p = Path(src)
        lines.append(f"| {p.as_posix()} | {p.suffix.lower().lstrip('.')} |")

    (kb_dir / "kb_manifest.md").write_text("\n".join(lines), encoding="utf-8")

def clear_raw_docs(kb_dir: str | Path):
    kb_dir = Path(kb_dir)
    keep = {"chroma.sqlite3", "kb_manifest.md"}

    for p in kb_dir.iterdir():
        if p.is_file() and p.name not in keep:
            p.unlink()   