import os
import pandas as pd
from abc import ABC, abstractmethod

import weaviate.classes as wvc
from weaviate.client import WeaviateClient
from weaviate.classes.config import Configure

import constants
from utils import make_properties, generate_uuid5, _row_exists


class BaseCollection(ABC):
    name = None
    data_file = None
    properties = []
    references = []

    vector_config = Configure.Vectors.text2vec_huggingface(
        model=constants.HF_MODEL,
    )
    generative_config = Configure.Generative.google(
        project_id=constants.PROJECT_ID,
        model_id="gemini-2.0-flash",
    )

    @classmethod
    def create(cls, client: WeaviateClient):
        print(f"Creating collection: '{cls.name}'...")
        client.collections.create(
            name=cls.name,
            vector_config=cls.vector_config,
            generative_config=cls.generative_config,
            properties=make_properties(cls.properties),
        )

    @classmethod
    @abstractmethod
    def build_properties(cls, row):
        raise NotImplementedError

    @classmethod
    def add_references(cls, client: WeaviateClient):
        if not cls.references:
            return
        collection = client.collections.get(cls.name)

        for reference in cls.references:
            try:
                collection.config.add_reference(reference)
            except Exception as e:
                print(f"Warning: Could not add reference {reference.name}: {e}")

    @classmethod
    def populate(cls, client: WeaviateClient):
        if not cls.data_file or not os.path.exists(cls.data_file):
            print(f"No data file for '{cls.name}', Skipping population...")
            return

        print(f"Populating collection '{cls.name}'...")
        collection = client.collections.get(cls.name)
        df = pd.read_csv(cls.data_file)

        objs = []
        for _, row in df.iterrows():
            row_objs = cls.build_properties(row)
            if isinstance(row_objs, dict):
                row_objs = [row_objs]

            for obj in row_objs:
                objs.append(
                    wvc.data.DataObject(
                        uuid=obj["uuid"],
                        properties=obj["properties"],
                        references=obj.get("references") or None,
                    )
                )

        if objs:
            collection.data.insert_many(objs)
        else:
            print(f"No objects to insert into '{cls.name}'")


class ReviewsCollection(BaseCollection):
    name = "Reviews"
    data_file = "movies.csv"
    properties = [
        ("body", wvc.config.DataType.TEXT, None),
    ]
    references = []

    @classmethod
    def build_properties(cls, row):
        objs = []
        for c in [1, 2, 3]:
            col = f"Critic Review {c}"
            text = row.get(col)

            if _row_exists(text):
                review_uuid = generate_uuid5(text.strip())
                objs.append(
                    {
                        "uuid": review_uuid,
                        "properties": {"body": text.strip()},
                    }
                )
        return objs

class SynopsisCollection(BaseCollection):
    name = "Synopsis"
    data_file = "movies.csv"
    properties = [
        ("body", wvc.config.DataType.TEXT, lambda row: row["Synopsis"]),
    ]
    references = [
        wvc.config.ReferenceProperty(
            name="forMovie",
            target_collection="Movies",
        ),
    ]

    @classmethod
    def build_properties(cls, row):
        props = {name: fn(row) for name, _, fn in cls.properties}

        references = {}
        movie_id = row.get("ID")
        if _row_exists(movie_id):
            references["forMovie"] = generate_uuid5(movie_id)
        else:
            references["forMovie"] = None

        return {
            "uuid": generate_uuid5(row.get("Synopsis")),
            "properties": props,
            "references": references,
        }


class MoviesCollection(BaseCollection):
    name = "Movies"
    data_file = "movies.csv"
    properties = [
        ("title", wvc.config.DataType.TEXT, lambda row: row["Movie Title"]),
        ("description", wvc.config.DataType.TEXT, lambda row: row["Description"]),
        ("rating", wvc.config.DataType.NUMBER, lambda row: float(row["Star Rating"])),
        ("movie_id", wvc.config.DataType.INT, lambda row: int(row["ID"])),
        ("year", wvc.config.DataType.INT, lambda row: int(row["Year"])),
        ("director", wvc.config.DataType.TEXT, lambda row: row["Director"]),
    ]
    references = [
        wvc.config.ReferenceProperty(
            name="hasReviews",
            target_collection=ReviewsCollection.name,
        ),
        wvc.config.ReferenceProperty(
            name="hasSynopsis",
            target_collection=SynopsisCollection.name,
        ),
    ]

    @classmethod
    def build_properties(cls, row):
        props = {name: fn(row) for name, _, fn in cls.properties}
        references = {}

        synopsis_text = row.get("Synopsis")
        if _row_exists(synopsis_text):
            references["hasSynopsis"] = generate_uuid5(synopsis_text.strip())
        else:
            references["hasSynopsis"] = None

        review_uuids = []
        for c in [1, 2, 3]:
            col = f"Critic Review {c}"
            text = row.get(col)
            if _row_exists(text):
                review_uuids.append(generate_uuid5(text.strip()))
        references["hasReviews"] = review_uuids or None

        return {
            "uuid": generate_uuid5(row.get("ID")),
            "properties": props,
            "references": references,
        }
