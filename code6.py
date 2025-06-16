from langchain.embeddings import AzureOpenAIEmbeddings
import os
import json
from dotenv import load_dotenv
load_dotenv()

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("TEXT_EMBEDDING_API_BASE"),
    openai_api_key=os.getenv("AZURE_API_KEY"),
    deployment=os.getenv("TEXT_EMBEDDING_API_NAME"),
    chunk_size=1000,
) # type:ignore
text = open("files/marwadi_university.txt", "r").read()

#text = "This is a sample text to be embedded using Azure OpenAI embeddings."
embeddings= embedding_model.embed_query(text)
print("Length of embeddings:", len(embeddings))  # Should be 1536 for text-embedding-3-small


with open("files/marwadi_university_embeddings.json", "w") as json_file:
    json.dump(embeddings, json_file)
print("Embeddings saved to files/marwadi_university_embeddings.json")
