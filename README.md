# Code Search Engine

Search your codebase using plain English. This tool chunks your code intelligently and lets you search it like Google.

## What Does This Do?

1. Reads all Python files from your codebase
2. Breaks them into smart chunks (functions, classes stay together)
3. Stores them in a vector database (Qdrant)
4. Lets you search using natural language like "authentication function" or "database connection"

## Prerequisites

1. **Python 3.12+** - Check with `python --version`
2. **Docker** - For running Qdrant locally
3. That's it.

## Quick Start

### 1. Install Dependencies

```bash
pip install chonkie qdrant-client sentence-transformers fastembed transformers tree-sitter-language-pack tiktoken
```

### 2. Start Qdrant (Vector Database)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Configure `main.py`

Edit these lines in `main.py`:

```python
CODEBASE_DIR = "/path/to/your/code"     # Change this to your codebase
COLLECTION_NAME = "my_code"              # Name for your search index
SHOULD_INGEST = True                     # Set to True first time
```

### 4. Run It

```bash
python main.py
```

**First run**: Set `SHOULD_INGEST = True` to load your code into the database  
**After that**: Set `SHOULD_INGEST = False` to just search

## How to Search

Edit the search query at the bottom of `main.py`:

```python
search_codebase(handshake, "your search query here", limit=3)
```

Examples:
- `"authentication function"`
- `"database connection"`
- `"API endpoint for users"`
- `"error handling code"`

## What's in Each File?

- **main.py** - Main script, run this
- **utils.py** - Helper functions
- **pyproject.toml** - Dependencies list