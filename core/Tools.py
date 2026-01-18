from core.Agent import Agent
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import create_retriever_tool
import math

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
    docs = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)
    db = Chroma.from_documents(chunks, HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
        ))
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
    return docs