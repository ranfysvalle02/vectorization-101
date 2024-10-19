import numpy as np

def trigram_hash(word, vocab_size=8000):
    """Create a trigram hash for a given word."""
    word = "_" + word + "_"
    embeddings = []
    
    for i in range(len(word) - 2):
        trigram = word[i:i+3]
        hashed_trigram = hash(trigram) % vocab_size
        embeddings.append(hashed_trigram)
    
    return embeddings

# Padding embeddings to ensure they have the same length
def pad_embedding(embedding, length=10):
    """Pad or truncate embedding to a fixed length."""
    if len(embedding) < length:
        # Pad with zeros if the embedding is shorter
        embedding += [0] * (length - len(embedding))
    elif len(embedding) > length:
        # Truncate if the embedding is longer
        embedding = embedding[:length]
    return embedding

# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# In-memory vector store (using a dictionary)
vector_store = {}

def add_to_store(word, embedding):
    """Add word and its trigram embedding to the in-memory store."""
    # Pad the embedding before storing
    padded_embedding = pad_embedding(embedding)
    vector_store[word] = np.array(padded_embedding)

def search_store(query_embedding, top_k=5):
    """Search for the top_k most similar embeddings using cosine similarity."""
    # Pad the query embedding
    padded_query_embedding = np.array(pad_embedding(query_embedding))
    similarities = {}
    
    for word, stored_embedding in vector_store.items():
        similarity = cosine_similarity(padded_query_embedding, stored_embedding)
        similarities[word] = similarity
    
    # Sort by highest similarity and return top_k results
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

# Example usage:

# Add some noisy embeddings to the store
add_to_store("cat", trigram_hash("cat"))
add_to_store("bat", trigram_hash("bat"))

# Query with another similar word (creating potential noise)
query_embedding = trigram_hash("hat")  # "hat" will overlap with "cat", "bat", and "rat"
results = search_store(query_embedding, top_k=3)

# Output the results
print("Top similar embeddings:", results)


add_to_store("rat", trigram_hash("rat"))


# Add some noisy embeddings to the store
add_to_store("sat", trigram_hash("cat"))
add_to_store("mat", trigram_hash("bat"))

# Query with another similar word (creating potential noise)
query_embedding = trigram_hash("hat")  # "hat" will overlap with "cat", "bat", and "rat"
results = search_store(query_embedding, top_k=3)

# Output the results
print("Top similar embeddings:", results)

# 'bat' goes from 0.94 to disappearing from the list


# Add some noisy embeddings to the store
add_to_store("mongodb", trigram_hash("mongodb"))
add_to_store("vectorstore", trigram_hash("vectorstore"))

# Query with another similar word (creating potential noise)
query_embedding = trigram_hash("hat")  # "hat" will overlap with "cat", "bat", and "rat"
results = search_store(query_embedding, top_k=3)

# Output the results
print("Top similar embeddings:", results)

# results stay the same, because the added words are not individually similar to the query

"""
Top similar embeddings: [('cat', 0.9961965520021488), ('bat', 0.9449372524399832)]
Top similar embeddings: [('cat', 0.9961965520021488), ('sat', 0.9961965520021488), ('rat', 0.9691060512487673)]
"""
