from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine

def calculate_semantic_diversity(texts, measure='average'):
    """
    Calculate semantic diversity for a set of texts.
    
    Parameters:
    - texts (list of str): The set of texts to be evaluated.
    - measure (str): Type of semantic diversity measure: 'average' for average pairwise distance,
                     or 'centroid' for distance to centroid.
                     
    Returns:
    - float: The semantic diversity score.
    """
    # Load the pre-trained Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert texts into embeddings
    embeddings = model.encode(texts)
    
    # Calculate semantic diversity based on the specified measure
    if measure == 'average':
        # Calculate average pairwise cosine distance
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distance = cosine(embeddings[i], embeddings[j])
                distances.append(distance)
        semantic_diversity = np.mean(distances)
    elif measure == 'centroid':
        # Calculate mean cosine distance from each embedding to the centroid
        centroid = np.mean(embeddings, axis=0)
        distances = [cosine(embedding, centroid) for embedding in embeddings]
        semantic_diversity = np.mean(distances)
    else:
        raise ValueError("Measure must be 'average' or 'centroid'")
    
    return semantic_diversity
