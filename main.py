import os
import time
from datetime import datetime
from typing import List

import numpy as np
from decouple import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from psycopg2.extensions import register_adapter, AsIs
from psycopg2.pool import SimpleConnectionPool

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY")

DB_CONFIG = {
    "dbname": "temporal-rag",
    "user": "root",
    "password": "rootpwd",
    "host": "localhost",
    "port": 5432,
}

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

connection_pool = SimpleConnectionPool(10, 20, **DB_CONFIG)


class VectorRecord:
    def __init__(self, embedding: List[float], timestamp: datetime, text: str):
        self.embedding = embedding
        self.timestamp = timestamp
        self.text = text

    def __repr__(self):
        return (
            f"VectorRecord(embedding={self.embedding}, "
            f"timestamp={self.timestamp}, text={self.text})"
        )

    def to_dict(self):
        return {"embedding": self.embedding, "timestamp": self.timestamp, "text": self.text}


def create_db():
    with connection_pool.getconn() as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS vectors")
        cursor.execute(
            """
            CREATE TABLE vectors (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                text TEXT,
                embedding vector(768)
            );
            CREATE INDEX ON vectors USING ivfflat (embedding vector_l2_ops);
            """
        )
        conn.commit()


def adapt_numpy_array(numpy_array):
    return AsIs(numpy_array.tolist())


register_adapter(np.ndarray, adapt_numpy_array)


def insert_vector(vector_record: VectorRecord):
    with connection_pool.getconn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
                INSERT INTO vectors (embedding, timestamp, text)
                VALUES (%s, %s, %s)
            """,
            (vector_record.embedding, vector_record.timestamp, vector_record.text),
        )
        conn.commit()


def query_vectors(query_embedding: List[float], k=5):
    with connection_pool.getconn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                    SELECT * FROM vectors
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """,
                (query_embedding, k),
            )
            rows = cursor.fetchall()

    if not rows:
        return []

    return [VectorRecord(embedding=row[3], timestamp=row[1], text=row[2]) for row in rows]


def add_text_to_database(text: str):
    embedding = embeddings_model.embed_query(text)
    timestamp = datetime.now()
    vector_record = VectorRecord(embedding, timestamp, text)
    insert_vector(vector_record)


def generate_response(query_text: str):
    query_embedding = embeddings_model.embed_query(query_text)
    results = query_vectors(query_embedding)

    if not results:
        return "No relevant information found."

    curr_time = time.time()

    formatted_data = "\n".join(
        [
            f"Time: {format_time_ago(curr_time - result.timestamp.timestamp())}, Context: {result.text}"
            for result in results
        ]
    )

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=formatted_data, question=query_text)

    print(f"Prompt: {prompt}")

    llm = GoogleGenerativeAI(model="gemini-1.5-flash")
    response_text = llm.invoke(prompt)

    return response_text


def format_time_ago(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    elif seconds < 86400:
        return f"{int(seconds // 3600)} hours ago"
    elif seconds < 604800:
        return f"{int(seconds // 86400)} days ago"
    elif seconds < 2629746:
        return f"{int(seconds // 604800)} weeks ago"
    elif seconds < 31556952:
        return f"{int(seconds // 2629746)} months ago"
    elif seconds >= 31556952:
        return f"{int(seconds // 31556952)} years ago"
    else:
        return "A long time ago"


def populate_db():
    create_db()

    # Cats
    add_text_to_database("Cats are small carnivorous mammals.")
    add_text_to_database("Cats are domesticated animals.")
    add_text_to_database("Cats are often kept as pets.")
    add_text_to_database("Cats are known for their agility.")

    # dogs
    add_text_to_database("Dogs are domesticated mammals.")
    add_text_to_database("Dogs are often kept as pets.")
    add_text_to_database("Dogs are known for their loyalty.")
    add_text_to_database("Dogs are known for their sense of smell.")


def main():
    populate_db()

    response = generate_response("What are cats? when did you learn it ?")

    print(f"Response: {response}")


if __name__ == "__main__":
    main()

connection_pool.closeall()
