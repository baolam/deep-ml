from collections import Counter

def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    corpus = corpus.lower()
    word = word.lower()

    tokens = corpus.split()
    vocab = Counter(tokens)

    return vocab.get(word) / len(tokens)