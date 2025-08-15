import utils

client = utils.connect_to_demo_db()

movies = client.collections.get("Movie")

response = movies.generate.near_text(
    query="stellar",
    limit=3,
    single_prompt="""
    Summarize the description:
    {description} for this movie {title}.
    """,
)
