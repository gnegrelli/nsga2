import numpy as np


# Function that evaluates fitness of a given individual
def fitness(crom, n, x1_l, x1_h, x2_l, x2_h):

    import numpy as np

    x1 = (crom[0]/(10.**n))*(x1_h - x1_l) + x1_l
    x2 = (crom[1]/(10.**n))*(x2_h - x2_l) + x2_l

    return x1*np.sin(10*np.pi*x1) + x2*np.cos(3*np.pi*(x2**2))


# Limits of x
xmin = -1
xmax = 2

# Limits of y
ymin = -.5
ymax = 1.8

# Size of population
pop_size = 100

# Maximum number of generations
max_generation = 100

# Maximum number of consecutive generations with no change in population
max_unchange = 10

# Precision
digits = 6

# Mutation tax
mutax = 0.1

# Current generation
gen = 0

# Number of consecutive generations with no change in population
unchange = 0

# List of individuals
specimens = []

# Creating initial population with unique individuals
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

print "Initial Generation"
print 30*"-"
for spec in specimens:
    print "%d: %s" % (specimens.index(spec) + 1, spec[1])
print "\n\n"

while gen < max_generation and unchange < max_unchange:

    # Initiate child as one specimen that already exists, forcing code to enter the following 'while' loop
    child = specimens[0][1]

    # This loop prevents parent to generate a child that already exists on the population
    while child in zip(*specimens)[1]:

        # List of selected parents
        parent = []

        # Selection process via tournament
        while len(parent) < 2:

            # Choosing champions to duel. The while loop prevents to choose the same champion twice
            champion1 = np.random.randint(pop_size)
            while champion1 in parent:
                champion1 = np.random.randint(pop_size)

            champion2 = np.random.randint(pop_size)
            while champion2 == champion1 or champion2 in parent:
                champion2 = np.random.randint(pop_size)

            # Tournament
            if specimens[champion1][0] > specimens[champion2][0]:
                parent.append(specimens[champion1][1])
            elif specimens[champion2][0] > specimens[champion1][0]:
                parent.append(specimens[champion2][1])
            elif np.random.rand() >= 0.5:
                parent.append(specimens[champion1][1])
            else:
                parent.append(specimens[champion2][1])

        gene_p = [bin(parent[0][0])[2:].zfill(20) + bin(parent[0][1])[2:].zfill(20),
                  bin(parent[1][0])[2:].zfill(20) + bin(parent[1][1])[2:].zfill(20)]

        # Crossover to generate child
        if len(gene_p[0]) == len(gene_p[1]):
            pivot = np.random.randint(len(gene_p[0]))
            if np.random.rand() > 0.5:
                child = list(gene_p[0][:pivot] + gene_p[1][pivot:])
            else:
                child = list(gene_p[1][:pivot] + gene_p[0][pivot:])
        else:
            "DEU PAU!!!!!"
            exit()

        # print "Pre-mutation:\t", ''.join(child)

        # Mutation
        for i in range(len(child)):
            if np.random.rand() <= mutax:
                if child[i] == '0':
                    child[i] = '1'
                else:
                    child[i] = '0'

        # print "Post-mutation:\t", ''.join(child)

        # Convert child from binary to int
        child = (int('0b' + ''.join(child[:20]), 2), int('0b' + ''.join(child[20:]), 2))

    # Add child to last position of population
    specimens.append([fitness(child, digits, xmin, xmax, ymin, ymax,), child])

    # If child is better than worst individual, add it to population
    if specimens[-1][0] >= specimens[-2][0]:
        specimens = sorted(specimens, reverse=True)[:pop_size]
        unchange = 0
    else:
        specimens = specimens[:pop_size]
        unchange += 1

    # Add counter of generations
    gen += 1

    print "Generation #%d" % gen
    print 30*"-"
    for spec in specimens:
        if spec[1] == child:
            print "\x1b[;32;49m%d: %.4f, %.4f \x1b[0m" % (specimens.index(spec) + 1, spec[1][0]/(10.**digits)*(xmax-xmin) + xmin, spec[1][1]/(10.**digits)*(ymax-ymin) + ymin)
        else:
            print "%d: %.4f, %.4f" % (specimens.index(spec) + 1, spec[1][0]/(10.**digits)*(xmax-xmin) + xmin, spec[1][1]/(10.**digits)*(ymax-ymin) + ymin)
    print "\n\n"

print "Best value: %.4f, %.4f" % (specimens[0][1][0]/(10.**digits)*(xmax - xmin) + xmin, specimens[0][1][1]/(10.**digits)*(ymax - ymin) + ymin)
print "Objective function: %.6f" % specimens[0][0]
