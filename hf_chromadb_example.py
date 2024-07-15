# Save embeddings via HF Transformers to a vector database via ChromaDB


import chromadb
import numpy as np
# from torch import tensors
import torch
from transformers import AutoTokenizer, BertTokenizer
from chromadb import PersistentClient

# Specify the path where you want the database files to be stored
path = "./chroma_vector_db"

# Create a PersistentClient with the specified path to save client to your local disk.
# This is only recommended for development not production
chroma_client = PersistentClient(path=path)

########## How can i change the default model ("all-MiniLM-L6-v2") to another to change the dimensions of embeddings from 384 to whatever I want? 
########## How can I view the collection? Ans: Chroma DB Viewer (see Github Repo)


# Create a client instance
# chroma_client = chromadb.Client()

# # Delete collection if necessary
# chroma_client.delete_collection("my_vector_collection_1")


'''
By default, the model "all-MiniLM-L6-v2" embeds the data into 384 dimensions, 
so in order to change the encoding function, you will need to use a different model 
or tokenizer. Pass the parameter "embedding_function" and set to a custom
python function that inputs a sequence and converts it to a list 
(no pytorch or tensorflow) embedding with the size of the new dimension. 
'''

class CustomEmbeddingFunction:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input):
        sequence = input
        print(sequence)
        # Use hugging face transformers to compute embeddings for models BERT
        input_info = self.tokenizer(sequence, padding=True, truncation=True, max_length=10)
        input_ids = input_info['input_ids']
        print('input_ids: ', input_ids)
        print(tokenizer.decode(input_ids[0]))
        return input_ids

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
custom_embedding_func = CustomEmbeddingFunction(tokenizer)

# Create the collection of vectors
chroma_collection = chroma_client.create_collection(name="my_vector_collection_2", embedding_function=custom_embedding_func, get_or_create=True)

# If the embedding_function parameter is not instatiated, then chromadb only allows tensors to be size Nx384 due to default model "all-MiniLM-L6-v2"
# chroma_collection = chroma_client.create_collection(name="my_vector_collection_2", get_or_create=True)



'''
Add text sequences to the collection.

This will automatically tokenize, embed and index automatically.
This all includes...
    - the documents to be vectorized 
    - metadata which are descriptive words for each document 
    - index represents the unique indices for each document

# DISCLAIMER: it seems that chromadb only allows tensors to be size Nx384 due to preprocessing pertaining to the model "all-MiniLM-L6-v2"
'''
chroma_collection.add(
    documents=["I am learning about vector databases and other rag systems plus a good amount of vector math which I always love to do.", "Do transformers include attention layers?"], 
    metadatas=[{"Topic": "vector database", "Doc Type": "statement"}, {"Topic": "transformers", "Doc Type": "question"}], 
    ids=["id1", "id2"])





# Or you can add your own embeddings via Numpy, Pytorch or Tensorflow, 

# # Numpy
# embeddings_arr = [np.random.rand(10).tolist(), np.random.rand(10).tolist()]
# # Remember to transform arrays to list types before adding it to the collection 
# embedding_list = embeddings_arr
# # The .add() collection methods adds the sequence embeddings 
# chroma_collection.add(
#     embeddings=embedding_list,
#     documents=["Is this is a document for other categories?", "This second document talks about transformers"],
#     metadatas=[
#         {"Topic": "other", "Doc Type": "question"},
#         {"Topic": "transformers", "Doc Type": "statement"}
#         ],
#     ids=["id5", "id6"]
#     )



# # Pytorch
# # Example 1
# chroma_collection.add(
#     # embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
#     embeddings=[torch.rand(10).tolist(), torch.rand(10).tolist()],
#     documents=["This is a document", "This is another document"],
#     metadatas=[{"source": "my_source"}, {"source": "my_source"}],
#     ids=["id3", "id4"]
# )


# # Example 2
# embedding_list_2 = map(lambda x: [y*2 for y in x], embedding_list)
# embeddings_tensor = torch.tensor(embedding_list).tolist()

# chroma_collection.add(
#     embeddings=embeddings_tensor,
#     documents=["Who wants to hear a joke today?", "What are the six main trig functions used?"],
#     metadatas=[
#         {"Topic": "random", "Doc Type": "question"},
#         {"Topic": "math", "Doc Type": "question"}
#         ],
#     ids=["id7", "id8"]
#     )


'''
Query the collection to find the top N most similar results. Parameters include
  query_texts: a list of query(s) 
  n: the number of similar results to be outputted.
'''
chroma_query = chroma_collection.query(
    query_texts = ['Transformers use multi-head attention layers vector databases.'],
    n_results=1
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
    )


# OR include query_embeddings
chroma_query_2 = chroma_collection.query(
    # query_embeddings=torch.rand(384).tolist(),
    query_embeddings=torch.rand(10).tolist(),
    n_results=3,
    # where={"metadata_field": "is_equal_to_this"},
    # where_document={"$contains":"search_string"}
)





if __name__ == '__main__':
    print('Start')
    print("----------------")
    print(chroma_query)
    print("----------------")
    print(chroma_query_2)
    print("----------------")
    print('Fin')

