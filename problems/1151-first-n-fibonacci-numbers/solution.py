def first_n_fibonacci(n):
    # Return a list of the first n Fibonacci numbers
    ans = []
    prev, curr = 0, 1
    if n >= 1:
        ans.append(0)
    if n >= 2:
        ans.append(1)
    
    for _ in range(2, n):
        prev, curr = curr, prev + curr
        ans.append(curr)
    
    return ans