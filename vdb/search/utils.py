import os
import subprocess
from dotenv import load_dotenv

import weaviate
from weaviate.client import WeaviateClient

load_dotenv()


def refresh_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error refereshing token: {result.stderr}")
        return None
    return result.stdout.strip()


def connect_to_demo_db() -> WeaviateClient:
    token = refresh_token()

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("DEMO_WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("DEMO_WEAVIATE_RO_KEY")),
        headers={
            # "X-Openai-Api-Key": os.getenv("OPENAI_APIKEY"),
            # "X-Goog-Studio-Api-Key": os.getenv("GOOGLE_APIKEY"),
            # "X-Goog-Vertex-Api-Key": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            "X-Goog-Vertex-Api-Key": token,
        },
    )
    return client


def main():
    client = connect_to_demo_db()

    try:
        assert client.is_ready()

        movies = client.collections.get("Movie")
        response = movies.query.near_text(query="time travel", limit=1)
        assert len(response.objects) == 1
        print(f"Response: {response.objects}")
        print("Client setup successfully")
    finally:
        client.close()


if __name__ == "__main__":
    main()
