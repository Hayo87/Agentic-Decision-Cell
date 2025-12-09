from tools.BaseTool import BaseTool
from pathlib import Path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

class KnowledgeBaseTool(BaseTool):
    def __init__(self, agent: str, k: int = 3):

        # Store
        self.agent = agent
        self.k = k

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
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
        self.index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)


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
    