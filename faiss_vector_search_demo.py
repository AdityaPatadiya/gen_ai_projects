from sentence_transformers import SentenceTransformer
import os
from langchain.document_loaders import TextLoader
import faiss

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def read_file(file_path):
    text = ""
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            text = file.read()
    return text


def store_vector(loader):
    vector = model.encode([loader], convert_to_numpy=True)
    dimension=vector.shape[1]

    index=faiss.IndexFlatL2(dimension)
    index.add(vector)
    faiss.write_index(index, "files/fais_vector.index")
    print("Vector stored successfully.")
    return vector


def search(query):
    index = faiss.read_index("files/fais_vector.index")
    query_vector = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, 1)
    return distances, indices

if os.path.exists("files/fais_vector.index"):
    pass
else:
    file_path = "files/marwadi_university.txt"
    loader = read_file(file_path)
    vector = store_vector(loader)

query = "what farmer do?"
distance, indices = search(query)
print(f"Distances: {distance}")
print(f"Indices: {indices}")
