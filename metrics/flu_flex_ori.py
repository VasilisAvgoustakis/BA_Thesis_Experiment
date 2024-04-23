import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.tokenize import sent_tokenize

def calculate_fluency(text):
    # Tokenizing the text
    words = word_tokenize(text)
    # Optionally, remove stopwords to focus on meaningful words
    stopwords = nltk.corpus.stopwords.words('english')
    meaningful_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords]
    
    # Counting unique words
    word_counts = Counter(meaningful_words)
    # Total unique words
    unique_words = len(word_counts)
    
    return unique_words


def calculate_flexibility(text, category_keywords):
    sentences = sent_tokenize(text)
    categories = set()

    for sentence in sentences:
        for category, keywords in category_keywords.items():
            if any(keyword in sentence.lower() for keyword in keywords):
                categories.add(category)
    
    return len(categories)

# Example categories and keywords
category_keywords = {
    'technology': ['computer', 'internet', 'software'],
    'nature': ['tree', 'river', 'mountain']
}


def calculate_originality(text, reference_corpus_freq, threshold=0.01):
    words = word_tokenize(text)
    unique_words = set(word.lower() for word in words if word.isalpha())
    
    # Count how many words appear less frequently than the threshold in the reference corpus
    original_words = sum(1 for word in unique_words if reference_corpus_freq.get(word, 0) < threshold)
    
    # Normalize by the number of unique words to get a measure of originality
    if unique_words:
        return original_words / len(unique_words)
    return 0

# Mocking a reference corpus frequency dictionary
reference_corpus_freq = {
    'the': 0.1, 'and': 0.09, 'to': 0.08, 'computer': 0.001, 'internet': 0.002, 'mountain': 0.005
}
