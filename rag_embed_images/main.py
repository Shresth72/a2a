from semantic import emded_images, retrieve_similar_images, retrieve_dog_similar_to_cat


def main():
    vector_store = emded_images("./images/*.jpeg")
    retriever = vector_store.as_retriever()

    retrieve_similar_images(retriever, "./images/cat_1.jpeg")
    retrieve_dog_similar_to_cat(retriever)


if __name__ == "__main__":
    main()
