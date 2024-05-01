import re
import nltk
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from scipy.stats import gmean

def read_data(file_path):
    """Read stories from the given file path where each story is separated by two newlines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    stories = content.split('\n\n')
    return stories

# Define the function to preprocess the text
def preprocess_text(text):
    # Remove all special characters and retain words
    return re.sub(r'[^\w\s]', '', text)

# Define the function to calculate MS-Jaccard
def calculate_ms_jaccard(real_stories, synthetic_stories, n=2, pseudocount=0.5):
    """
    Calculate the MS-Jaccard similarity based on n-grams across all stories,
    smoothing results with a pseudocount to prevent zero penalties.
    Normalization is per story.
    """
    def ngram_freq_per_sentence(stories, n):
        ngram_freqs = []
        for story in stories:
            text = preprocess_text(story).lower()
            ngram_counts = Counter(ngrams(word_tokenize(text), n))
            num_sentences = len(text.split('.'))  # Example split on sentences
            ngram_freqs.append({ngram: count / num_sentences for ngram, count in ngram_counts.items()})
        return ngram_freqs

    # Get normalized n-gram frequencies for both real and synthetic stories
    real_ngram_freqs = ngram_freq_per_sentence(real_stories, n)
    synthetic_ngram_freqs = ngram_freq_per_sentence(synthetic_stories, n)

    # Aggregate the n-gram frequencies across all stories
    real_freq = Counter()
    synthetic_freq = Counter()
    
    for freqs in real_ngram_freqs:
        real_freq.update(freqs)
    for freqs in synthetic_ngram_freqs:
        synthetic_freq.update(freqs)
    
    # Calculate the MS-Jaccard score for each n-gram and take the geometric mean
    scores = []
    all_ngrams = set(real_freq.keys()) | set(synthetic_freq.keys())
    for ngram in all_ngrams:
        real_count = real_freq.get(ngram, pseudocount)
        synthetic_count = synthetic_freq.get(ngram, pseudocount)
        min_count = min(real_count, synthetic_count)
        max_count = max(real_count, synthetic_count)
        score = min_count / max_count
        scores.append(score)
    
    # Compute the geometric mean of the scores
    ms_jaccard_score = gmean(scores) if scores else 0

    return ms_jaccard_score

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
