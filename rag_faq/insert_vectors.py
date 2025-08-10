from datetime import datetime

import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

vec = VectorStore()

df = pd.read_csv("./data/rag_sample_qas_from_kis.csv", sep=";")


def prepare_record(row):
    content = (
        f"Question: {row['sample_question']}\nAnswer: {row['sample_ground_truth']}"
    )

    embedding = vec.get_emdedding(content)
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "topic": row["ki_topic"],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )


records_df = df.apply(prepare_record, axis=1)

vec.create_tables()
vec.create_index()
vec.upsert(records_df)
