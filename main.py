import os
import time

from chonkie import CodeChunker, QdrantHandshake, AutoEmbeddings
import tiktoken

from utils import compute_time_elapsed, get_extension

tokenizer = tiktoken.get_encoding("gpt2")
embeddings = AutoEmbeddings.get_embeddings(model="all-MiniLM-L6-v2")


def read_python_files(directory, language_extension=".py", exclude_dirs=None):
    """
    Recursively read all Python files from a directory and return their contents as strings.

    Args:
        :param directory: Root directory to scan
        :param language_extension: File extension to filter (default: ".py")
        :param exclude_dirs: List of directory names to exclude (default: ['.git', '__pycache__', 'venv', '.venv', 'node_modules'])
    Returns:
        list: List of strings, where each string is the content of a Python file
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'venv', '.venv', '.idea', 'build', 'dist']

    codes = []

    for root, dirs, files in os.walk(directory):
        # Filter out excluded directories in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(language_extension):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                        codes.append(code_content)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    return codes

@compute_time_elapsed()
def chunk_and_ingest_codebase(
        directory: str,
        language: str = "python",
        collection_name: str = "codebase_chunks",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str = None,
        chunk_size: int = 2048,
        should_ingest: bool = False,
        debug_mode: bool = False
):
    """
    Chunk an entire codebase and ingest it into Qdrant using Chonkie's Handshake.

    Args:
        directory: Path to the codebase directory
        language: Programming language (python, javascript, etc.)
        collection_name: Name for the Qdrant collection
        qdrant_url: URL of Qdrant server (or cloud URL)
        qdrant_api_key: API key for Qdrant Cloud (optional for local)
        chunk_size: Maximum tokens per chunk
        should_ingest: will govern to ingest into vector search engine or not
        debug_mode: will govern to print logs in debug mode
    """

    # Step 1: Initialize the CodeChunker
    chunker = CodeChunker(
        language=language,
        chunk_size=chunk_size,
        # include_nodes=True,
        # tokenizer="gpt2"  # or "gpt2" or "cl100k_base" (GPT-4 encoding) etc
    )

    codes = read_python_files(directory=directory, language_extension=get_extension(language))

    batch_chunks = chunker.chunk_batch(codes)

    # print the chunks for verification
    # for doc_chunks in batch_chunks:
    #     for chunk in doc_chunks:
    #         print(f"Chunk: {chunk.text}")

    handshake = QdrantHandshake(
        url=qdrant_url,
        api_key=qdrant_api_key,
        embedding_model=embeddings,
        collection_name=collection_name,
    )

    start_time = time.time()
    if should_ingest:
        for doc_chunks in batch_chunks:
            for chunk in doc_chunks:
                if debug_mode:
                    print(f"Chunk: {chunk.text}")
                handshake.write(chunk)

    end_time = time.time()
    print(f"\nIngestion Time: {end_time - start_time:.2f} seconds")

    return handshake


# Example Usage
def search_codebase(handshake, query, limit):
    print(f"\nSearching for: '{query}'")
    print("-" * 80)

    results = handshake.search(query=query, limit=limit)

    for i, result in enumerate(results, 1):
        print(f"\n[Result {i}] Score: {result['score']:.4f}")
        print(f"File: {result.get('filepath', 'Unknown')}")
        print(f"Chunk {result.get('chunk_index', 'N/A')} | Tokens: {result.get('token_count', 'N/A')}")
        print(f"\nCode:\n{result['text']}...")
        print("-" * 80)

    return results


if __name__ == "__main__":
    # Configuration
    CODEBASE_DIR = "/Users/pavanmantha/Downloads/ai_code"  # Your codebase path
    LANGUAGE = "python"
    COLLECTION_NAME = "code_collection_2"

    # For local Qdrant
    QDRANT_URL = "http://localhost:6333"
    QDRANT_API_KEY = "th3s3cr3tk3y"

    # For Qdrant Cloud (uncomment and add your credentials)
    # QDRANT_URL = "https://your-cluster.qdrant.io"
    # QDRANT_API_KEY = "your-api-key"

    # Chunk and ingest the codebase
    handshake = chunk_and_ingest_codebase(
        directory=CODEBASE_DIR,
        language=LANGUAGE,
        collection_name=COLLECTION_NAME,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        chunk_size=2048,
        should_ingest=False,
        debug_mode=False
    )

    # Example searches
    print("\n" + "=" * 80)
    print("SEARCH EXAMPLES")
    print("=" * 80)

    # Search for specific functionality
    # example1: def get_deepseek_v32_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    # example2: self.scheduler.get_kv_connector()
    # example3: logger.info(f\"Selecting retriever {result.ind}: {result.reason}.\")
    search_codebase(handshake, "QdrantVectorStore", limit=3)
    