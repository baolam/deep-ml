def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	import numpy as np
    a = np.array(a)
    b = np.array(b)
    try:
        return (a @ b).tolist()
    except:
        return -1