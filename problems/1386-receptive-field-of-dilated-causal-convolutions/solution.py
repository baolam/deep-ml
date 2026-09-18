def receptive_field(kernel_sizes, dilations):
    """R = 1 + sum((k - 1) * d) over layers. Returns int."""
    # Your code here
    R = 1
    for k, l in zip(kernel_sizes, dilations):
        R += (k - 1) * l
    return R

def wavenet_receptive_field(n_layers, kernel_size, n_blocks=1):
    """n_blocks blocks of dilations 1, 2, 4, ..., 2**(n_layers-1). Returns int."""
    # Your code here
    # $$R = 1 + B \times (k - 1)(2^n - 1)$$
    return 1 + n_blocks * (kernel_size - 1) * (2 ** n_layers - 1)
    

def causal_padding(kernel_size, dilation):
    """Left padding that keeps the sequence length: (kernel_size - 1) * dilation."""
    # Your code here
    return (kernel_size - 1) * dilation
