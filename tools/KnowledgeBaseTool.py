from tools.BaseTool import BaseTool
from pathlib import Path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter

class KnowledgeBaseTool(BaseTool):
    """
    KnowledgeBaseTool is a tool for querying an agent-specific knowledge base using vector embeddings and similarity search.

    This tool loads documents from a predefined directory for a given agent,
    creates embeddings using a configurable HuggingFace model, and builds a vector index to
    retrieve the most relevant content based on a query.

    Attributes:
        agent (str): Name of the agent whose knowledge base is loaded.
        k (int): Number of top results to retrieve per query (default: 3).
        embedding_model (str): Choice of embedding model for document encoding.
            Options:
                - "classic": High-quality, slightly slower (109M parameters)
                - "small": Fast, moderate quality (102M parameters)
                - "normal": Balanced performance and size (334M parameters)
                - "large": Best quality, large model (1.3B parameters)
                - "multilingual": Supports multiple languages, long texts (up to 8192 tokens)
        chunk_size (int): Size of text chunks for splitting documents (default: 512 tokens).
        chunk_overlap (int): Number of overlapping tokens between chunks (default: 100).
        index (VectorStoreIndex | None): Vector index built from agent documents for retrieval.
            None if no knowledge base is found.

    Methods:
        supports(action: str, target: str, agent: str) -> bool:
            Returns True if this tool can handle the given action for the agent.
        run(content: str) -> str:
            Executes a query against the knowledge base and returns the most relevant snippet.
    """

    def __init__(self, agent: str, k: int = 3, embedding_model = "classic", chunk_size=512, chunk_overlap=100):

        # Store
        self.agent = agent
        self.k = k
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize the knowledge base
        project_root = Path(__file__).resolve().parents[1]
        kb_path = project_root / "agents" / "kb" / agent

        # Check for errors
        if not kb_path.exists() or not any(kb_path.iterdir()):
            print(f"[KnowledgeBaseTool] No knowledge base found for agent '{agent}' at {kb_path}")
            self.index = None
            return
        
        # Load the documents 
        docs = SimpleDirectoryReader(str(kb_path)).load_data()

        # Load embedding model - multiple options available
        try:
            if embedding_model == "classic":
                Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
            elif embedding_model == "small":
                Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            elif embedding_model == "normal":
                Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
            elif embedding_model == "large":
                Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
            elif embedding_model == "multilingual":
                Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
            else:
                raise ValueError("Unknown Embedding Model")
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            raise

        # Dynamic configuration for splitting text from external documents
        Settings.text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Create Vector Index
        self.index = VectorStoreIndex.from_documents(docs)

    def supports(self, action: str, target: str, agent: str) -> bool:
        """Return True if this tool can handle the given request."""
        return (
            action == "search"
            and target == "KnowledgeBase"
            and agent == self.agent
        )
    
    def run(self, content: str) -> str:
        """Execute the tool and return an observation."""
        if self.index is None:
            return "No information available."

        retriever = self.index.as_retriever(similarity_top_k=self.k)
        nodes = retriever.retrieve(content)

        snippets = [node.get_content() for node in nodes]
        best = snippets[0] if snippets else ""

        return best  
    