import re

def selective_normalize_spaced_text(text):
    # Define a regex pattern to find spaced-out words specifically
    # This pattern assumes spaced-out words have at least two spaces between the letters.
    spaced_out_pattern = r'\b(?:\w\s){2,}\w\b'

    # Function to reconstruct a single spaced-out word
    def reconstruct_word(match):
        word = match.group(0)
        # Remove all spaces within the word
        return word.replace(' ', '')

    # Apply reconstruction only to spaced-out words
    text = re.sub(spaced_out_pattern, reconstruct_word, text)

    # Normalize spaces between words (reduce multiple spaces to a single space)
    text = re.sub(r'\s{2,}', ' ', text)

    return text