from dotenv import load_dotenv

from weaviate.exceptions import WeaviateBaseError

from database import Database
from models import MoviesCollection, ReviewsCollection, SynopsisCollection

load_dotenv()


def main():
    db = Database()

    try:
        db.init_collections([ReviewsCollection, SynopsisCollection, MoviesCollection])

        # results = db.query_generate(
        #     MoviesCollection,
        #     query="stellar",
        #     limit=1,
        #     prompt="""
        #         Summarize the description:
        #         {description} for this movie {title}.
        #         """,
        # )
        # for r in results:
        #     print(f"{r['generated']}\n")
    except WeaviateBaseError as e:
        print(f"Weaviate error occured: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
