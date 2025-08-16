import main
import weaviate.classes as wvc

client = main.connect_to_db()

movies = client.collections.get("Movie")

# ------- RAG -------
response = movies.generate.near_text(
    query="stellar",
    limit=3,
    single_prompt="""
    Summarize the description:
    {description} for this movie {title}.
    """,
)

# ------- HYBRID -------
response = movies.query.hybrid(
    query="stellar",
    limit=3,
    alpha=0.9,  # More vector search
    return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
)

# ------- FILTER -------
filter = wvc.query.Filter.by_property("year").greater_or_equal(
    1990
) & wvc.query.Filter.by_property("description").like("space")

response = movies.query.near_text(
    query="science fiction",
    limit=2,
    filters=filter,
)

for o in response.objects:
    movie_id = o.properties["movie_id"]
    movie_title = o.properties["title"]
    movie_year = o.properties["year"]

    print(f"ID: {movie_id}, {movie_title}, year: {movie_year}")
    print(o.properties["description"][:50] + "...\n")

client.close()
