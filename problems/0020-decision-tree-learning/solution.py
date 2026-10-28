import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    # Your code here
    if not labels:
        return 0.0
    
    total = len(labels)
    counts = Counter(labels)

    entropy = 0.0

    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    # Your code here
    if not examples:
        return 0.0
    
    parent_labels = [ ex[target_attr] for ex in examples ]
    parent_entropy = calculate_entropy(parent_labels)

    total = len(examples)

    subsets = {}
    for ex in examples:
        val = ex[attr]
        subsets.setdefault(val, []).append(ex)
    
    weighted_entropy = 0.0
    for subset in subsets.values():
        subset_labels = [ ex[target_attr] for ex in subset ]
        weighted_entropy += (len(subset) / total) * calculate_entropy(subset_labels)
    
    return parent_entropy - weighted_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    # Your code here
    if not examples:
        return ""
    
    labels = [ ex[target_attr] for ex in examples ]
    counts = Counter(labels)

    return sorted(counts.keys(), key=lambda c: (-counts[c], str(c)))[0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    # Your code here
    if not examples:
        return {}
    
    labels = [ ex[target_attr] for ex in examples ]
    if len(set(labels)) == 1:
        return labels[0]
    
    if not attributes:
        return majority_class(examples, target_attr)
    
    best_attr = None
    best_gain = -1.0

    for attr in attributes:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    
    tree = {
        best_attr : {}
    }
    remaining_attributes = [a for a in attributes if a != best_attr]
    unique_values = sorted(set(ex[best_attr] for ex in examples))

    for val in unique_values:
        sub_examples = [ ex for ex in examples if ex[best_attr] == val ]
        if not sub_examples:
            tree[best_attr][val] = majority_class(examples, target_attr)
        else:
            tree[best_attr][val] = learn_decision_tree(sub_examples, remaining_attributes, target_attr)

    return tree