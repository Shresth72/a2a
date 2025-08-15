import os
import json
import subprocess
from dotenv import load_dotenv

import weaviate
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


def upload_to_db(client: WeaviateClient):
    print("Creating and populating 'Movies' collection...")
    client.collections.delete(name="Movies")
    movies = client.collections.create(
        name="Movies",
        vector_config=Configure.Vectors.text2vec_huggingface(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
        generative_config=Configure.Generative.google(
            project_id=os.getenv("PROJECT_ID"),
            model_id="gemini-2.0-flash",
        ),
    )

    with open("../movies.json", "r") as f:
        data = json.load(f)

    error_count = 0
    uploaded_count = 0

    with movies.batch.dynamic() as batch:
        for d in data:
            batch.add_object(
                properties={
                    "title": d["title"],
                    "description": d["description"],
                    "rating": d["rating"],
                    "movie_id": d["movie_id"],
                    "year": d["year"],
                    "director": d["director"],
                }
            )
            if batch.number_errors > error_count:
                error_count = batch.number_errors
            else:
                uploaded_count += 1

            if error_count > 10:
                print("Batch import failed — too many errors, stopping.")
                break

    print(f"Uploaded successfully: {uploaded_count} out of {len(data)}")

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
