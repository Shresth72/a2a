import os
import json
import uuid
import pandas as pd

import weaviate.classes as wvc
from weaviate.client import WeaviateClient
from weaviate.classes.config import Configure

import constants
from utils import make_properties


class BaseCollection:
    __name__ = None
    data_file = None
    properties = []

    vector_config = Configure.Vectors.text2vec_huggingface(
        model=constants.HF_MODEL,
    )
    generative_config = Configure.Generative.google(
        project_id=constants.PROJECT_ID,
        model_id="gemini-2.0-flash",
    )

    @classmethod
    def create(cls, client: WeaviateClient):
        print(f"Creating collection: '{cls.__name__}'...")
        client.collections.create(
            name=cls.__name__,
            vector_config=cls.vector_config,
            generative_config=cls.generative_config,
            properties=make_properties(cls.properties),
        )

    @classmethod
    def populate(cls, client: WeaviateClient):
        if not cls.data_file or not os.path.exists(cls.data_file):
            print(f"No data file for '{cls.__name__}', skipping population")
            return

        print(f"Populating collection '{cls.__name__}'...")
        collection = client.collections.get(cls.__name__)

        df = pd.read_csv(cls.data_file)
        objs = []
        for _, row in df.iterrows():
            data = cls.map_row(row)
            objs.append(
                wvc.data.DataObject(
                    uuid=uuid.uuid4(),
                    properties=data,
                )
            )

        collection.data.insert_many(objs)

    @staticmethod
    def map_row(row):
        return row.to_dict()


class MoviesCollection(BaseCollection):
    __name__ = "Movies"
    properties = [
        ("title", wvc.config.DataType.TEXT),
        ("description", wvc.config.DataType.TEXT),
        ("rating", wvc.config.DataType.NUMBER),
        ("movie_id", wvc.config.DataType.INT),
        ("year", wvc.config.DataType.INT),
        ("director", wvc.config.DataType.TEXT),
    ]
    data_file = "movies.csv"

    @staticmethod
    def map_row(row):
        return {
            "title": row["Movie Title"],
            "description": row["Description"],
            "rating": float(row["Star Rating"]),
            "movie_id": int(row["ID"]),
            "year": int(row["Year"]),
            "director": row["Director"],
        }


class ReviewsCollection(BaseCollection):
    __name__ = "Reviews"
    properties = [
        ("body", wvc.config.DataType.TEXT),
    ]
    data_file = "movies.csv"

    @staticmethod
    def map_row(row):
        return {
            "body": row["Critic Review 1"],
        }
