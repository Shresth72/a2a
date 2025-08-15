import os
import json
import uuid
import subprocess
from dotenv import load_dotenv

import weaviate
import weaviate.classes as wvc
from weaviate.client import WeaviateClient
from weaviate.classes.config import Configure
from weaviate.exceptions import WeaviateBaseError

load_dotenv()


def refresh_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error refreshing token: {result.stderr}")
        return None
    return result.stdout.strip()


def connect_to_demo_db() -> WeaviateClient:
    token = refresh_token()

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        headers={
            "X-HuggingFace-Api-Key": os.getenv("HF_API_KEY"),
            "X-Goog-Vertex-Api-Key": token,
        },
    )
    return client


def make_properties(fields):
    return [
        wvc.config.Property(name=name, data_type=data_type)
        for name, data_type in fields
    ]


def upload_to_db(client: WeaviateClient):
    client.collections.delete(name="Movies")

    print("Creating and populating 'Movies' collection...")

    movies = client.collections.create(
        name="Movies",
        vector_config=Configure.Vectors.text2vec_huggingface(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
        generative_config=Configure.Generative.google(
            project_id=os.getenv("PROJECT_ID"),
            model_id="gemini-2.0-flash",
        ),
        properties=make_properties(
            [
                ("title", wvc.config.DataType.TEXT),
                ("description", wvc.config.DataType.TEXT),
                ("rating", wvc.config.DataType.NUMBER),
                ("movie_id", wvc.config.DataType.INT),
                ("year", wvc.config.DataType.INT),
                ("director", wvc.config.DataType.TEXT),
            ]
        ),
    )

    with open("../movies.json", "r") as f:
        movie_data = json.load(f)

    movie_objs = list()
    for i, data in enumerate(movie_data):
        movie_uuid = uuid.uuid4()
        props = {
            "title": data["title"],
            "description": data["description"],
            "rating": data["rating"],
            "movie_id": data["movie_id"],
            "year": data["year"],
            "director": data["director"],
        }

        data_obj = wvc.data.DataObject(
            uuid=movie_uuid,
            properties=props,
        )
        movie_objs.append(data_obj)

    movies.data.insert_many(movie_objs)
    return movies


def main():
    client = connect_to_demo_db()

    try:
        assert client.is_ready()

        movies = client.collections.get("Movies")
        if movies.exists():
            print("Using existing 'Movies' collection.")
        else:
            print("'Movies' collection not found. Uploading new collection.")
            movies = upload_to_db(client)

        # response = movies.query.near_text(query="space", limit=2)
        response = movies.generate.near_text(
            query="stellar",
            limit=2,
            single_prompt="""
            Summarize the description:
            {description} for this movie {title}.
            """,
        )
        for obj in response.objects:
            print(obj.properties["title"])
            print(obj.generated)
            print()

    except WeaviateBaseError as e:
        print(f"Weaviate error occurred: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
