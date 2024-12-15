# Temporal RAG System

A Retrieval-Augmented Generation (RAG) system that maintains temporal awareness of when information was added to the knowledge base. Built with PostgreSQL (pgvector), LangChain, and Google's Generative AI.

## About

This system implements a RAG architecture with the following features:
- Vector similarity search using pgvector
- Temporal tracking of when information was added
- Integration with Google's Generative AI for embeddings and text generation
- Connection pooling for efficient database operations
- Time-aware context formatting in responses

## Installation

### Dependencies

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Create a .env file in the project root and add your Google API key:

```text
GOOGLE_API_KEY=your_api_key_here
```

### PostgreSQL with pgvector
Install PostgreSQL if you haven't already:

```bash
# Ubuntu/Debian
sudo apt-get install postgresql

# macOS with Homebrew
brew install postgresql
```

Install pgvector:

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-14-pgvector

# macOS with Homebrew
brew install pgvector
```

Create a database and user for the project:

```bash
createdb temporal-rag
createuser -P -d -R -S -l temporal-rag
```

Grant permissions to the user:

```bash
psql temporal-rag
GRANT USAGE ON SCHEMA public TO temporal-rag;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO temporal-rag;
```

## Usage

To run the application, execute the following command:

```bash
python main.py
```