import utils
import weaviate.classes as wvc

client = utils.connect_to_demo_db()

movies = client.collections.get("Movie")

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
