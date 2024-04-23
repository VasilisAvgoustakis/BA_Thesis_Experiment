import re
import nltk
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter

def read_data(file_path):
    """Read stories from the given file path where each story is separated by two newlines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    stories = content.split('\n\n')
    return stories

def calculate_jaccard_similarity(stories, n=2):
    """Calculate the mean Jaccard similarity based on n-grams across all pairs of stories."""
    all_ngrams = [set(ngrams(word_tokenize(story.lower()), n)) for story in stories]
    scores = []
    for i in range(len(all_ngrams)):
        for j in range(i + 1, len(all_ngrams)):
            inter = all_ngrams[i].intersection(all_ngrams[j])
            union = all_ngrams[i].union(all_ngrams[j])
            scores.append(len(inter) / len(union) if union else 0)
    return np.mean(scores)

def calculate_feature_based_similarity(stories, model_name='bert-base-uncased'):
    """Calculate the feature-based similarity using embeddings from a pre-trained BERT model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for story in stories:
            inputs = tokenizer(story, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = model(**inputs.to(model.device))
            embeddings.append(outputs.last_hidden_state.mean(1))
    embeddings = torch.stack(embeddings).squeeze(1)
    similarity_matrix = cosine_similarity(embeddings.cpu().numpy())
    # Exclude self-similarities
    np.fill_diagonal(similarity_matrix, 0)
    return np.mean(similarity_matrix[np.triu_indices(len(stories), 1)])
