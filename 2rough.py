import csv
import matplotlib.pyplot as plt
import umap
import warnings
import os
from src import config as C

from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


def read_file(file_name: str) -> dict:
    """
    Read a csv file and return a dictionary.

    Parameters
    ----------
    file_name: str
        The csv file has columns: name and description.

    Return
    ------
    people: dict{str: str}
        dictionary {name: description}
    """
    people = {}
    with open(file_name, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            name, description = row
            people[name] = description
    return people


def sentence_embedding(model_name: str, people: dict) -> dict:
    """
    Embedding descriptions of people.

    Parameters
    ----------
    model_name: str
        Model name of SentenceTransformer package. E.g.: entence-transformers/all-MiniLM-L6-v2
    people: dict{str: str}
        dictionary {name: description}

    Return
    ------
    person_embeddings: {str: list[float]}
        dictionary {name: description_embedding}
    """
    person_embeddings = {}
    model = SentenceTransformer(model_name)
    for person in people.keys():
        person_embeddings[person] = model.encode(people[person])
    return person_embeddings


def umap_visualization(person_embeddings: dict) -> None:
    """
    Visualize people's embeddings using UMAP

    Parameters
    ----------
    person_embeddings: {str: list[float]}
        dictionary {name: description_embedding}
    """
    umap_model = umap.UMAP(random_state=42)
    umap_vectors = umap_model.fit_transform(list(person_embeddings.values()))

    plt.figure(figsize=(20, 10))
    plt.scatter(umap_vectors[:, 0], umap_vectors[:, 1])
    for i, person in enumerate(person_embeddings):
        plt.annotate(person, (umap_vectors[i, 0], umap_vectors[i, 1]))

    plt.show()

def make_pipeline(file_name: str, model_name: str, visualization: bool = False) -> dict:
    """
    Combine methods read_file(), sentence_embedding() and umap_visualization() into one pipleline.

    Parameters
    ----------
    file_name: str
    model_name: str
    visualization: bool

    Return
    ------
    person_embeddings: {str: list[float]}
        dictionary {name: description_embedding}
    """
    people = read_file(file_name)
    person_embeddings = sentence_embedding(model_name, people)
    if visualization:
        umap_visualization(person_embeddings)

    return person_embeddings


def similarity_ranking(person_embeddings: dict, name: str) -> list[list]:
    """
    Given the embeddings of people's description and a person, return a list of cosine similarities
    and cosine similarity rankings of that person to other people, sorted alphabetically by name.

    Parameters
    ----------
    person_embeddings: {str: list[float]}
        dictionary {name: description_embedding}
    name: str
        name of the person from whom to calculate cosine similarities with other people

    Return
    ------
    top_matches: list[list[str, float, int]]
        [[name, cosine_similarity, rank]] ranked alphabetically by name
    """
    # calculate cosine similarity
    top_matches = []
    for person in person_embeddings.keys():
        top_matches.append([person, 1 - cosine(person_embeddings[name], person_embeddings[person])])

    # add similarity ranking
    top_matches.sort(key=lambda x: -x[1])  # sort by cosine similarity
    for i, match in enumerate(top_matches):
        match.append(i)

    # sort by name
    top_matches = top_matches[1:]
    top_matches.sort(key=lambda x: x[0])
    return top_matches


def compare_models(file_name: str, model_1: str, model_2: str, person_name: str) -> None:
    """
    Compare two models from SentenceTransformer.
    """
    print(f"Comparing Models..")
    tm1 = similarity_ranking(make_pipeline(file_name, model_1), person_name)
    tm2 = similarity_ranking(make_pipeline(file_name, model_2), person_name)

    x = [i for i in range(len(tm1))]
    cosine_diff = [tm2[i][1] - tm1[i][1] for i in range(len(tm1))]
    rank_diff = [tm1[i][2] - tm2[i][2] for i in range(len(tm1))]

    plt.subplot(2, 1, 1)
    plt.scatter(x, cosine_diff, s=8)
    plt.axhline(y=0, color="red", linestyle="-")
    plt.ylabel("Cosine Difference")
    for i in range(len(tm1)):
        plt.annotate(tm1[i][0][:6], (x[i], cosine_diff[i]), fontsize="8")

    plt.subplot(2, 1, 2)
    plt.scatter(x, rank_diff, s=8)
    plt.axhline(y=0, color="red")
    plt.ylabel("Ranks gained")
    for i in range(len(tm1)):
        plt.annotate(tm1[i][0][:6], (x[i], rank_diff[i]), fontsize="8")

    plt.suptitle(
        f"Difference in cosine similarity and ranking of everyone vs {person_name} for {model_1} and {model_2}", size=16
    )
    plt.savefig(os.path.join(C.BASE_RESULTS_PATH , "model_comparison.png"))
    print(f"Figure Saved Succesfully!!")


if __name__ == "__main__":
    compare_models(C.CLASSMATES_DATA_PATH, C.MINI_MODEL_PATH, C.MPNET_MODEL_PATH, "Greg Kirczenow")

