from collections import Counter

def count_common_unique(list1, list2):
    # list1: list of strings
    # list2: list of strings
    # return an integer
    s1 = set(list1)
    s2 = set(list2)
    c1 = Counter(list1)
    c2 = Counter(list2)

    cnt = 0
    for word in list(s1 & s2):
        if c1[word] == 1 and c2[word] == 1:
            cnt += 1
    
    return cnt