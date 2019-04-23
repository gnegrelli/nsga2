import numpy as np


# Function that evaluates fitness of a given individual
def fitness(crom, n, x1_l, x1_h, x2_l, x2_h):

    import numpy as np

    x1 = (crom[0]/(10.**n))*(x1_h - x1_l) + x1_l
    x2 = (crom[1]/(10.**n))*(x2_h - x2_l) + x2_l

    return x1*np.sin(10*np.pi*x1) + x2*np.cos(3*np.pi*(x2**2))


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

# Precision
digits = 6

# List of individuals
specimens = []

# Creating unique individuals
for i in range(pop_size):
    a = np.random.randint(10**digits)
    b = np.random.randint(10**digits)

    # This loop prevents creating repeated individuals
    while [0, (a, b)] in specimens:
        a = np.random.randint(10**digits)
        b = np.random.randint(10**digits)

    # Add new individual to population
    specimens.append([0, (a, b)])

# Fitness calculation
for spec in specimens:
    spec[0] = fitness(spec[1], digits, xmin, xmax, ymin, ymax,)

# Sorting specimens for best to worst
specimens.sort(reverse=True)


