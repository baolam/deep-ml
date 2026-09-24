def allocator_ops(num_blocks, operations):
    # num_blocks: int, total number of fixed-size blocks
    # operations: list of tuples like ("alloc",) or ("free", idx)
    # return: list of results (int for alloc, bool for free)
    results = []

    allocated, free = [False] * num_blocks, []
    next_untouched = 0

    for op in operations:
        if op[0] == 'alloc':
            if free:
                idx = free.pop()
                allocated[idx] = True
                results.append(idx)
            elif next_untouched < num_blocks:
                idx = next_untouched
                allocated[idx] = True
                results.append(idx)
                next_untouched += 1
            else:
                results.append(-1)
        else:
            idx = op[1]

            if 0 <= idx < num_blocks and allocated[idx]:
                allocated[idx] = False
                free.append(idx)
                results.append(True)
            else:
                results.append(False)


    return results