import utils
import weaviate.classes as wvc

client = utils.connect_to_demo_db()

movies = client.collections.get("Movie")

response = movies.query.hybrid(
    query="stellar",
    limit=3,
    alpha=0.9,  # More vector search
    return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
)
