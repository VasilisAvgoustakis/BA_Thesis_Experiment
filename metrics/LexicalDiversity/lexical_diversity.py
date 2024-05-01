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
from nltk.tokenize import word_tokenize

def calculate_self_bleu(texts):
    """
    Calculate the Self-BLEU score for a set of texts.
    
    Parameters:
    - texts (list of str): The set of generated texts to be evaluated.
    
    Returns:
    - float: The average Self-BLEU score of the texts.
    """
    scores = []
    
    # Ensure that inputs are cleaned and non-empty
    cleaned_texts = [text.strip() for text in texts if text.strip()]
    
    # Check if there are fewer than two valid texts
    if len(cleaned_texts) < 2:
        return 0


    for i, candidate in enumerate(texts):
        if not candidate.strip():  # Skip empty candidates
            continue
        
        # Consider all other texts as references for the current candidate text
        references = [word_tokenize(texts[j].lower()) for j in range(len(texts)) if i != j and texts[j].strip()]
        candidate_tokens = word_tokenize(candidate.lower())
        
        if not references or not candidate_tokens:  # Skip if no valid data is available
            continue
        
        try:
            # Calculate the BLEU score for this text against all others
            score = sentence_bleu(references, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25))
            scores.append(score)
        except Exception as e:
            print(f"Error calculating BLEU for text index {i}: {e}")
            continue  # Handle possible errors during BLEU calculation
    
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
    #print("Total Tokens: ", len(tokens))
    
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

def calculate_mean_segmental_ttr(text, segment_size=50):
    """
    Calculate the mean segmental Type-Token Ratio (TTR) for the given text.
    
    Parameters:
    - text (str): The input text.
    - segment_size (int): The number of words in each text segment.
    
    Returns:
    - float: The mean TTR across all segments.
    """
    # Normalize the text to lowercase and split it into words
    tokens = text.lower().split()
    
    # Split the tokens into segments of the specified size
    segments = [tokens[i:i + segment_size] for i in range(0, len(tokens), segment_size)]
    
    # Compute the TTR for each segment
    ttrs = []
    for segment in segments:
        types = len(set(segment))  # Count unique words
        total_tokens = len(segment)  # Count total words
        ttr = types / total_tokens if total_tokens > 0 else 0  # Compute TTR for the segment
        ttrs.append(ttr)
        
    #print("cm_ttrs: ", ttrs)
    # Compute the mean TTR across all segments
    mean_ttr = sum(ttrs) / len(ttrs) if ttrs else 0
    return mean_ttr

def calculate_cumulative_ttr(text, increment=10, truncate_length=300):
    """
    Compute the cumulative TTR for the text.

    Parameters:
    - text (str): The input text.
    - increment (int): The word increment for computing cumulative TTRs.

    Returns:
    - List of tuples: Each tuple contains (number of words, TTR up to that point).
    """
    # Convert the text to lowercase and split into individual words
    tokens = text.lower().split()

    # If truncate_length is specified and less than the number of tokens, truncate the list of tokens
    if truncate_length is not None and len(tokens) > truncate_length:
        tokens = tokens[:truncate_length]

    # List to hold the cumulative TTR values
    cumulative_ttr_values = []

    # Iterate over the text in the specified word increments
    for i in range(increment, len(tokens) + increment, increment):
        # Ensure not to exceed total length of tokens
        i = min(i, len(tokens))
        #print(i)

        # Slice the tokens up to the current index and calculate TTR
        segment = tokens[:i]
        unique_words = len(set(segment))
        total_words = len(segment)
        ttr = unique_words / total_words if total_words > 0 else 0

        # Append the cumulative TTR value along with the number of words
        cumulative_ttr_values.append((total_words, ttr))

    return cumulative_ttr_values

import matplotlib.pyplot as plt

def plot_cumulative_ttr(cumulative_ttr_values):
    # Unpack the number of words and TTR values
    word_counts, ttr_values = zip(*cumulative_ttr_values)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(word_counts, ttr_values, marker='o')
    
    # Set the title and labels
    plt.title('Cumulative Type-Token Ratio (TTR)')
    plt.xlabel('Number of Words')
    plt.ylabel('TTR')
    
    # Show the grid
    plt.grid(True)
    
    # Show the plot
    plt.show()


def calculate_decremental_ttr_dynamic(text, segment_percentage=10):
    """
    Calculate the decremental Type-Token Ratio (TTR) where each segment is a specified percentage of the text length.

    Parameters:
    - text (str): The input text.
    - segment_percentage (int): The percentage of the total text length that each segment should represent.

    Returns:
    - list: A list containing the decremental TTR for each dynamic segment.
    """
    # Normalize text and split into tokens
    tokens = text.lower().split()
    # Determine dynamic segment size based on the specified percentage of the total text length
    segment_size = max(int(len(tokens) * (segment_percentage / 100)), 1)  # Ensure at least one word per segment

    segments = [tokens[i:i + segment_size] for i in range(0, len(tokens), segment_size)]
    
    unique_words = set()
    decremental_ttrs = []

    for segment in segments:
        # Count new unique words in the current segment
        new_unique_words = set(segment) - unique_words
        # Calculate the decremental TTR (number of new unique words / segment size)
        decremental_ttr = len(new_unique_words) / segment_size if segment_size > 0 else 0
        decremental_ttrs.append(decremental_ttr)
        # Update the set of all unique words encountered so far
        unique_words.update(segment)

    return decremental_ttrs

def plot_decremental_ttr(decremental_ttrs):
    """
    Plot the decremental TTR curve.

    Parameters:
    - decremental_ttrs (list): The decremental TTR values for each segment.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(decremental_ttrs) + 1), decremental_ttrs, marker='o')
    plt.title('Decremental Type-Token Ratio (TTR)')
    plt.xlabel('Segment Number')
    plt.ylabel('Decremental TTR')
    plt.grid(True)
    plt.show()