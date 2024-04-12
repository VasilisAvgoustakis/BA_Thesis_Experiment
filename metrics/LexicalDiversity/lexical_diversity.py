def calculate_distinct_n(text, n=2):
    """
    Calculate the Distinct-n metric for a given text. This metric evaluates the diversity of generated text 
    by counting the number of unique sequences of n words (n-grams).

    Parameters:
    - text (str): The text to be analyzed.
    - n (int): The length of the n-gram (e.g., 2 for Distinct-2).

    Returns:
    - float: The Distinct-n score as the proportion of unique n-grams to the total number of n-grams.
    """
    # Normalize the text to lowercase and split into words
    tokens = text.lower().split()
    
    # Generate n-grams from the list of tokens
    n_grams = [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    #print("Total n_grams: ", len(n_grams))
    # Calculate the number of unique n-grams
    unique_n_grams = len(set(n_grams))
    #print("Unique n_grams: ", unique_n_grams)
    # Calculate the total number of n-grams
    total_n_grams = len(n_grams)
    
    # Calculate the Distinct-n score
    distinct_n_score = unique_n_grams / total_n_grams if total_n_grams > 0 else 0
    
    return distinct_n_score


from nltk.translate.bleu_score import sentence_bleu

def calculate_self_bleu(texts):
    """
    Calculate the Self-BLEU score for a set of texts.
    
    Parameters:
    - texts (list of str): The set of generated texts to be evaluated.
    
    Returns:
    - float: The average Self-BLEU score of the texts.
    """
    scores = []
    for i, candidate in enumerate(texts):
        # Consider all other texts as references for the current candidate text
        references = [texts[j].split() for j in range(len(texts)) if i != j]
        candidate_tokens = candidate.split()
        # Calculate the BLEU score for this text against all others
        score = sentence_bleu(references, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        scores.append(score)
    
    # Calculate the average score across all texts
    average_score = sum(scores) / len(scores) if scores else 0
    return average_score

def calculate_ttr(text, truncate_length=None):
    """
    Calculate the Type-Token Ratio (TTR) for a given text.
    
    Parameters:
    - text (str): The input text.
    - truncate_length (int, optional): The length to which the text should be truncated before calculating TTR.
    
    Returns:
    - float: The TTR of the text.
    """
    # Normalize the text to lowercase to ensure uniformity
    text = text.lower()
    # Tokenize the text into words
    tokens = text.split()
    print("Total Tokens: ", len(tokens))
    
    # If truncate_length is specified and less than the number of tokens, truncate the list of tokens
    if truncate_length is not None and len(tokens) > truncate_length:
        tokens = tokens[:truncate_length]
    
    # Calculate the number of unique words (types)
    types = len(set(tokens))
    # Calculate the total number of words (tokens)
    total_tokens = len(tokens)
    
    # Avoid division by zero
    if total_tokens == 0:
        return 0
    
    # Calculate TTR
    ttr = types / total_tokens
    return ttr