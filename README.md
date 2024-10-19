# vectorization-101

### Trigram Hashing: Efficient Embeddings for Resource-Constrained NLP

In today's world, with large-scale models capturing much of the limelight, more efficient techniques like trigram hashing still play an important role in NLP, particularly in scenarios where memory efficiency and computational speed are essential. The code provided offers a hands-on demonstration of trigram hashing for embedding words and finding similar embeddings using cosine similarity.

Here, we break down how trigram hashing is used to create embeddings and why its flexibility and simplicity make it a valuable tool in 2024, especially in cases of noisy neighbors.

Trigram hashing and the Tokenization-Free approach are particularly useful in several scenarios:

1. **Resource-Constrained Environments**: Trigram hashing is computationally lightweight, making it ideal for environments where computational resources are limited. This includes devices with limited processing power or memory, such as mobile devices or embedded systems.

2. **Real-Time Applications**: The efficiency of trigram hashing makes it suitable for real-time applications where quick responses are required. This includes tasks like search, real-time translation, chatbots, and other interactive systems.

3. **Handling Noisy or Inconsistent Data**: The Tokenization-Free approach can handle variable-length inputs and is robust to misspellings and out-of-vocabulary terms. This makes it useful in scenarios where the input data may be noisy or inconsistent, such as user-generated content or transcriptions of spoken language.

#### The Process: Creating Trigram Embeddings
Trigram hashing works by breaking down words into overlapping trigrams (three-character sequences), then hashing these trigrams into fixed-size vocabulary indices. The method captures the structural patterns in the text without relying on a vast pre-built vocabulary, allowing for a fast, resource-efficient encoding process.

The `vocab_size=8000` does impose a sort of limitation like a context window, but it's more about how much **distinctiveness** you can capture in your hashed representations of trigrams. If the vocabulary size is too small (e.g., 1000), more trigrams will collide into the same hash value, reducing the ability to distinguish between them. This is somewhat analogous to how a small context window would limit how much information a model can capture from text at once.

In the provided code:
```python
def trigram_hash(word, vocab_size=8000):
    """Create a trigram hash for a given word."""
    word = "_" + word + "_"
    embeddings = []
    
    for i in range(len(word) - 2):
        trigram = word[i:i+3]
        hashed_trigram = hash(trigram) % vocab_size
        embeddings.append(hashed_trigram)
    
    return embeddings
```
The trigram hash function generates a list of hashed trigrams for any input word, ensuring that each word is represented as a sequence of integers that correspond to the trigrams' hashed values. This provides a compact and informative embedding for words, even when they are out-of-vocabulary or contain typos.

#### Embedding Flexibility and Padding
One key advantage of trigram hashing is its ability to handle variable-length inputs, which makes it highly flexible when dealing with noisy or inconsistent data. To further enhance this flexibility, embeddings can be padded or truncated to ensure consistency:
```python
def pad_embedding(embedding, length=10):
    """Pad or truncate embedding to a fixed length."""
    if len(embedding) < length:
        embedding += [0] * (length - len(embedding))
    elif len(embedding) > length:
        embedding = embedding[:length]
    return embedding
```
By padding or truncating embeddings, the code ensures that all embeddings are of a fixed length, which is crucial for efficient comparison between embeddings in vector spaces.

#### Cosine Similarity and Vector Stores
The heart of this system is the in-memory vector store, where words and their embeddings are stored and compared using cosine similarity:
```python
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)
```
Cosine similarity measures the angle between two vectors, offering a way to quantify the similarity between two words' embeddings. This is especially useful in vector stores where quick, approximate searches for similar words are required.

#### Addressing Noisy Neighbors
The example provided also illustrates how adding words to the vector store can influence search results, especially in cases of "noisy neighbors." As new words with similar trigram structures are added to the store, they may cause previously high-ranked words to drop out due to the introduction of new embeddings with similar trigram hashes. This highlights one of the challenges in vector embeddings—how to handle noise or near-duplicates in the data.

In the example, after adding new words like "mat" and "bat," the word "hat" becomes more closely associated with "bat" and "mat," which share similar trigram structures. These noisy neighbors can crowd out previously higher-ranked results:
```python
Top similar embeddings: [('bat', 0.9718733783459191), ('mat', 0.9718733783459191), ('cat', 0.9539424639949023)]
```
While the new words are not direct synonyms, the trigram overlaps make them appear similar in the context of the embedding space. This behavior is both a strength and a limitation of trigram hashing—its simplicity allows for quick embeddings, but it can also be sensitive to noise in certain applications.

#### The Power of Trigram Hashing in 2024
In 2024, with the surge of resource-heavy models like GPT and BERT, trigram hashing still offers an appealing alternative for certain applications. Its power lies in:
- **Efficiency**: Trigram hashing is computationally lightweight, making it ideal for environments where speed and memory are constrained.
- **Flexibility**: The method handles unseen words, misspellings, and out-of-vocabulary terms gracefully, providing robust performance even with noisy data.
- **Simplicity**: Trigram hashing doesn't require massive pre-training or large vocabulary lists, allowing it to scale easily and provide fast, real-time results in tasks like search, classification, and entity matching.

### Noisy Neighbors Section: Real-World Applications

The noisy neighbors effect, where similar trigram structures cause embeddings to collide, is particularly relevant in several real-world applications. For example, consider a **search engine** that uses embeddings to return the most relevant documents based on a user’s query. If several documents contain words with overlapping trigrams, like "cat," "bat," and "mat," the search results might group these terms too closely, potentially pushing more relevant but structurally dissimilar terms further down the results list. This could dilute the relevance of search results, especially in cases where precision is important, such as legal or medical document searches.

In **recommendation systems**, trigram hashing could lead to similar issues. Suppose a user is searching for products, like "black shoes," and similar-sounding but irrelevant products (like "blue shoes" or "blazer shirts") end up being recommended because their embeddings are too close in the vector space. This can reduce the accuracy of recommendations and lead to poor user experiences. In both cases, handling noisy neighbors is critical for maintaining the quality of the system’s output, requiring additional techniques such as refining the similarity threshold or implementing filtering mechanisms to mitigate these effects.

### Where Trigram Hashing Shines

Trigram hashing offers a versatile solution for several industries and products, particularly in situations where memory efficiency and real-time performance are essential. Here are some specific industries and examples:

1. **Healthcare**: In medical text processing, patient records often contain typos or abbreviations. Trigram hashing’s ability to handle noisy, out-of-vocabulary, or inconsistent input makes it ideal for **electronic health record (EHR) search systems**, where retrieving accurate information quickly is critical. It's also useful in **medical transcription services**, where the input may contain misspellings or unfamiliar terms.

2. **E-commerce**: For **product search engines** and **recommendation systems** in e-commerce platforms, trigram hashing can efficiently handle queries with typos or variations in product names. This is especially useful for global companies that need to accommodate different languages, as trigram hashing can assist in matching products across multiple languages without relying on an extensive, pre-built vocabulary.

3. **Embedded Systems**: In devices like **smartphones**, **wearables**, or **IoT devices**, where memory and processing power are limited, trigram hashing provides a lightweight method for performing tasks like **voice command recognition** or **real-time text classification**. For instance, a voice assistant on a smart speaker might rely on trigram embeddings to parse and respond to user commands efficiently, without requiring the heavy computational load of larger models.

By offering a memory-efficient, flexible solution, trigram hashing is particularly well-suited to industries where data is often noisy, real-time performance is needed, and computational resources are constrained.

