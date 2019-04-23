import numpy as np

print type(bin(10))

# Limits of x
xmin = -1
xmax = 2

# Limits of y
ymin = -.5
ymax = 1.8

# Size of population
pop_size = 10

# Maximum number of generations
gen_max = 10

# Current generation
gen = 0

specimens = []

for i in range(pop_size):
    a = np.random.randint(10.**6)
    b = np.random.randint(10.**6)
    while [0, (a, b)] in specimens:
        a = np.random.randint(10.**6)
        b = np.random.randint(10.**6)
    specimens.append([0, (a, b)])
