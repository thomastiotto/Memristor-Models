import numpy as np
import time

# time code block
start = time.time()

# code block
x = np.random.normal(0, 1, 1000000)

# time code block
end = time.time()
print('Pre-generation takes', end - start, 'seconds')

# time code block
start = time.time()

# code block
for i in range(1000000):
    x = np.random.normal(0, 1)

# time code block
end = time.time()
print('Iteratively generating takes', end - start, 'seconds')
