import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.tokenize import sent_tokenize

import numpy as np

nltk.download('stopwords')

def calculate_fluency(texts):
    scores = []

    for text in texts:
        # Tokenizing the text
        words = word_tokenize(text)
        # Optionally, remove stopwords to focus on meaningful words
        stopwords = nltk.corpus.stopwords.words('english')
        meaningful_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords]
        
        # Counting unique words
        word_counts = Counter(meaningful_words)
        # Total unique words
        unique_words = len(word_counts)
        scores.append(unique_words)
    
    return np.array(scores).mean()/100


def calculate_flexibility(texts, category_keywords):
    scores = []

    for text in texts:
        sentences = sent_tokenize(text)
        categories = set()

        for sentence in sentences:
            for category, keywords in category_keywords.items():
                if any(keyword in sentence.lower() for keyword in keywords):
                    categories.add(category)
        scores.append(len(categories))
    print(scores)
    return np.array(scores).mean()/100




def calculate_originality(texts, reference_corpus_freq, threshold=0.01):
    scores = []

    for text in texts:
        #print(text)
        words = word_tokenize(text)
        unique_words = set(word.lower() for word in words if word.isalpha())
        
        # Count how many words appear less frequently than the threshold in the reference corpus
        original_words = sum(1 for word in unique_words if reference_corpus_freq.get(word, 0) < threshold)
    
        # Normalize by the number of unique words to get a measure of originality
        if unique_words:
            scores.append(original_words / len(unique_words))
        else:
            scores.append(0)
    return np.array(scores).mean()


