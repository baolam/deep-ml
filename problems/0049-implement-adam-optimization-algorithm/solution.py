import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    t = 0
    thetat = x0
    mt = np.zeros(thetat.shape)
    vt = np.zeros(thetat.shape)

    for _ in range(num_iterations):
        t = t + 1
        gt = grad(thetat)
        mt = beta1 * mt + (1 - beta1) * gt
        vt = beta2 * vt + (1 - beta2) * gt ** 2

        mt_cab = mt / (1 - beta1 ** t)
        vt_cab = vt / (1 - beta2 ** t)

        thetat = thetat - learning_rate * mt_cab / (np.sqrt(vt_cab) + epsilon)
    
    return thetat