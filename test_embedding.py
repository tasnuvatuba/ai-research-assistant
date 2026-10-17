from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

sentence = (
    "Personalized federated learning adapts a model "
    "to individual clients."
)

vector = embeddings.embed_query(sentence)

print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])