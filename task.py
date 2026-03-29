import torch
#import numpy as np

print(torch.accelerator.current_accelerator().type)
print(torch.accelerator.is_available())

# np.random.seed(42)

# X = np.random.randint(0, 2, size=(100, 12))

# Y = np.random.randint(0, 2, size=(100, 2))


# np.savetxt('dataIn.txt', X, fmt='%d')

# np.savetxt('dataOut.txt', Y, fmt='%d')