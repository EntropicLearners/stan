"""
Segment and index lecture transcripts for semantic search.

Uses LlamaIndex to chunk transcripts, embed them with a HuggingFace model,
and build a persistent vector index. Supports querying with Ollama-based
LLM synthesis.

Pipeline:
    Transcript (.txt)
         │
    SentenceSplitter (512 tokens, 50 overlap)
         │
    Embed (BAAI/bge-large-en-v1.5)
         │
    VectorStoreIndex (persisted to disk)
         │
    Query → top-k retrieval → LLM synthesis (Ollama)

Usage:
    # Build index from a transcript
    from stan.lecture.segment import LectureIndex

    idx = LectureIndex()
    idx.build("stan/data/lectures/transcripts/lecture.txt")

    # Query
    response = idx.query("What did the professor say about entropy generation?")
    print(response)

    # CLI
    python -m stan.lecture.segment build transcript.txt
    python -m stan.lecture.segment query "entropy generation"
"""

from pathlib import Path

# Default parameters — matching ssearch conventions with adjustments for lectures
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE = 512       # larger than ssearch's 256 — lecture topics run longer
CHUNK_OVERLAP = 50     # proportional overlap
LLM_MODEL = "llama3.1:8B"
SIMILARITY_TOP_K = 15
PERSIST_DIR = "stan/data/lectures/index"


class LectureIndex:
    """Build and query a vector index over lecture transcripts."""

    def __init__(
        self,
        embed_model=EMBED_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        llm_model=LLM_MODEL,
        persist_dir=PERSIST_DIR,
        similarity_top_k=SIMILARITY_TOP_K,
    ):
        self.embed_model_name = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model = llm_model
        self.persist_dir = Path(persist_dir)
        self.similarity_top_k = similarity_top_k

        self._index = None
        self._embed_model = None
        self._llm = None

    def _get_embed_model(self):
        if self._embed_model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self._embed_model = HuggingFaceEmbedding(
                model_name=self.embed_model_name
            )
        return self._embed_model

    def _get_llm(self):
        if self._llm is None:
            from llama_index.llms.ollama import Ollama
            self._llm = Ollama(
                model=self.llm_model,
                request_timeout=360.0,
            )
        return self._llm

    def build(self, transcript_path, persist=True):
        """Build a vector index from a transcript file.

        Args:
            transcript_path: Path to a .txt transcript file
            persist: If True, save the index to disk
        """
        from llama_index.core import (
            Document,
            Settings,
            StorageContext,
            VectorStoreIndex,
        )
        from llama_index.core.node_parser import SentenceSplitter

        transcript_path = Path(transcript_path)
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        text = transcript_path.read_text()
        print(f"Loaded transcript: {transcript_path.name} "
              f"({len(text)} chars, ~{len(text.split())} words)")

        # Configure LlamaIndex settings
        Settings.embed_model = self._get_embed_model()
        Settings.llm = self._get_llm()

        # Create document with metadata
        doc = Document(
            text=text,
            metadata={
                "source": transcript_path.name,
                "type": "lecture_transcript",
            },
        )

        # Chunk the transcript
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator="\n",
        )
        nodes = splitter.get_nodes_from_documents([doc])
        print(f"Created {len(nodes)} chunks "
              f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

        # Build vector index
        print("Building vector index (embedding chunks)...")
        self._index = VectorStoreIndex(nodes)

        if persist:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._index.storage_context.persist(
                persist_dir=str(self.persist_dir)
            )
            print(f"Index persisted to {self.persist_dir}")

        return self._index

    def load(self):
        """Load a previously persisted index from disk."""
        from llama_index.core import (
            Settings,
            StorageContext,
            load_index_from_storage,
        )

        if not self.persist_dir.exists():
            raise FileNotFoundError(
                f"No persisted index at {self.persist_dir}. Run build() first."
            )

        Settings.embed_model = self._get_embed_model()
        Settings.llm = self._get_llm()

        storage_context = StorageContext.from_defaults(
            persist_dir=str(self.persist_dir)
        )
        self._index = load_index_from_storage(storage_context)
        print(f"Loaded index from {self.persist_dir}")
        return self._index

    def query(self, query_text):
        """Query the lecture index and return a synthesized response.

        Args:
            query_text: Natural language query

        Returns:
            LlamaIndex Response object with .response text and .source_nodes
        """
        if self._index is None:
            self.load()

        query_engine = self._index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
        )
        response = query_engine.query(query_text)
        return response

    def retrieve(self, query_text):
        """Retrieve matching chunks without LLM synthesis.

        Useful for inspecting what the index finds before synthesis.

        Args:
            query_text: Natural language query

        Returns:
            List of NodeWithScore objects
        """
        if self._index is None:
            self.load()

        retriever = self._index.as_retriever(
            similarity_top_k=self.similarity_top_k,
        )
        return retriever.retrieve(query_text)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and query a lecture transcript index"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build subcommand
    build_parser = subparsers.add_parser("build", help="Build index from transcript")
    build_parser.add_argument("transcript", help="Path to transcript .txt file")
    build_parser.add_argument(
        "--persist-dir", default=PERSIST_DIR,
        help=f"Directory to save index (default: {PERSIST_DIR})"
    )
    build_parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Chunk size in tokens (default: {CHUNK_SIZE})"
    )
    build_parser.add_argument(
        "--chunk-overlap", type=int, default=CHUNK_OVERLAP,
        help=f"Chunk overlap in tokens (default: {CHUNK_OVERLAP})"
    )

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query the lecture index")
    query_parser.add_argument("query", help="Natural language query")
    query_parser.add_argument(
        "--persist-dir", default=PERSIST_DIR,
        help=f"Directory of persisted index (default: {PERSIST_DIR})"
    )
    query_parser.add_argument(
        "--top-k", type=int, default=SIMILARITY_TOP_K,
        help=f"Number of chunks to retrieve (default: {SIMILARITY_TOP_K})"
    )
    query_parser.add_argument(
        "--retrieve-only", action="store_true",
        help="Show retrieved chunks without LLM synthesis"
    )

    args = parser.parse_args()

    if args.command == "build":
        idx = LectureIndex(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            persist_dir=args.persist_dir,
        )
        idx.build(args.transcript)

    elif args.command == "query":
        idx = LectureIndex(
            persist_dir=args.persist_dir,
            similarity_top_k=args.top_k,
        )

        if args.retrieve_only:
            nodes = idx.retrieve(args.query)
            print(f"\nRetrieved {len(nodes)} chunks:\n")
            for i, node in enumerate(nodes, 1):
                score = node.score
                text = node.node.get_content()[:200]
                print(f"[{i}] score={score:.4f}")
                print(f"    {text}...")
                print()
        else:
            response = idx.query(args.query)
            print(f"\n{response.response}")
            print(f"\n--- Sources ({len(response.source_nodes)} chunks) ---")
            for i, node in enumerate(response.source_nodes, 1):
                text = node.node.get_content()[:100]
                print(f"[{i}] score={node.score:.4f}: {text}...")


if __name__ == "__main__":
    main()
