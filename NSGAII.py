import numpy as np
import copy
import matplotlib.pyplot as plt


# Function that evaluates fitness of a given individual
def fitness(crom, n, x1_l, x1_h, x2_l, x2_h):

    import numpy as np

    x1 = (crom[0]/(2.**n))*(x1_h - x1_l) + x1_l
    x2 = (crom[1]/(2.**n))*(x2_h - x2_l) + x2_l

    return x1*np.sin(10*np.pi*x1) + x2*np.cos(3*np.pi*(x2**2)), x1**3 + x2**2


# Flags
print_gen = True
local_search = not True

# Limits of x
xmin = -1
xmax = 2

# Limits of y
ymin = -.5
ymax = 1.8

# Size of population
pop_size = 30

# Maximum number of generations
max_generation = 1000

# Maximum number of consecutive generations with no change in population
max_unchange = max(3, int(.02*max_generation))

# Precision
digits = 6

# Mutation tax
mutax = 0.1

# Crossover tax
crosstax = 0.7

# Number of bits needed
bits = int(np.ceil(np.log2(10**digits)))

# Current generation
gen = 0

# Number of consecutive generations with no change in population
unchange = 0

# List of individuals
specimens = []

# Creating initial population with unique individuals
for i in range(pop_size):
    a = np.random.randint(2**bits)
    b = np.random.randint(2**bits)

    # This loop prevents creating repeated individuals
    while [-1, 0, 0, (a, b)] in specimens:
        a = np.random.randint(2**bits)
        b = np.random.randint(2**bits)

    # Add new individual to population
    specimens.append([-1, 0, 0, (a, b), 99999, 99999])

# Fitness calculation
for spec in specimens:
    spec[1], spec[2] = fitness(spec[3], bits, xmin, xmax, ymin, ymax)

# Sorting specimens from best to worst
specimens.sort(reverse=True)

aux_specimens = []
tier = 0

# Ranking process
while specimens:

    f2 = np.array([x[2] for x in specimens])
    for i in range(len(specimens)):
        G = (f2 > f2[i])
        if np.all(G[:i]):
            specimens[i][0] = tier
            aux_specimens.append(specimens[i])

    tier += 1

    plt.scatter([x[1] for x in specimens], [x[2] for x in specimens])

    for i in range(len(specimens) - 1, -1, -1):
        if specimens[i] in aux_specimens:
            specimens.remove(specimens[i])

# Plot f1 versus f2 of every individual
plt.scatter([x[1] for x in specimens], [x[2] for x in specimens])
plt.show()

# Copy specimens ranked in tiers and organized
specimens = aux_specimens

# Calculate normalized crowding distance for each tier
for tiers in range(specimens[-1][0] + 1):
    aux = [x for x in specimens if x[0] == tiers]
    for i in range(1, len(aux) - 1):
        d = (np.sqrt(((aux[i][1] - aux[i - 1][1])/(aux[0][1] - aux[-1][1]))**2 +
                     ((aux[i][2] - aux[i - 1][2])/(aux[0][2] - aux[-1][2]))**2),
             np.sqrt(((aux[i][1] - aux[i + 1][1])/(aux[0][1] - aux[-1][1]))**2 +
                     ((aux[i][2] - aux[i + 1][2])/(aux[0][2] - aux[-1][2]))**2))
        aux[i][4], aux[i][5] = min(d), max(d)

if print_gen:
    print("Initial Generation")
    print(30*"-")
    for spec in specimens:
        print("%d: %s" % (specimens.index(spec) + 1, spec))
    print("\n\n")

while gen < max_generation and unchange < max_unchange:

    # In this loop, pairs of parents are selected to generate children, doubling the population
    while len(specimens) < 2*pop_size:

        # List of selected parents
        parent = []

        # Selection process via tournament
        while len(parent) < 2:

            # Choosing champions to duel. The while loop prevents to choose the same champion twice
            champ1 = np.random.randint(pop_size)
            while specimens[champ1][3] in parent:
                champ1 = np.random.randint(pop_size)

            champ2 = np.random.randint(pop_size)
            while champ2 == champ1 or specimens[champ2][3] in parent:
                champ2 = np.random.randint(pop_size)

            # Tournament
            # Check dominance
            if specimens[champ1][1] > specimens[champ2][1] and specimens[champ1][2] < specimens[champ2][2]:
                parent.append(specimens[champ1][3])
            elif specimens[champ1][1] < specimens[champ2][1] and specimens[champ1][2] > specimens[champ2][2]:
                parent.append(specimens[champ2][3])

            # Check smaller distance
            elif specimens[champ1][4] > specimens[champ2][4]:
                parent.append(specimens[champ1][3])
            elif specimens[champ1][4] < specimens[champ2][4]:
                parent.append(specimens[champ2][3])

            # Check higher distance
            elif specimens[champ1][5] > specimens[champ2][5]:
                parent.append(specimens[champ1][3])
            elif specimens[champ1][5] < specimens[champ2][5]:
                parent.append(specimens[champ2][3])

            # Random choice
            elif np.random.rand() >= 0.5:
                parent.append(specimens[champ1][3])
            else:
                parent.append(specimens[champ2][3])

        gene_p = [bin(parent[0][0])[2:].zfill(bits) + bin(parent[0][1])[2:].zfill(bits),
                  bin(parent[1][0])[2:].zfill(bits) + bin(parent[1][1])[2:].zfill(bits)]

        # Crossover to generate original child
        if len(gene_p[0]) == len(gene_p[1]):
            
            # Set pivot point. If pivot point is zero, one of the parents will be passed on
            if np.random.rand() <= crosstax:
                pivot = np.random.randint(len(gene_p[0]))
            else:
                pivot = 0
            
            # Create original child
            if np.random.rand() > 0.5:
                child = list(gene_p[0][:pivot] + gene_p[1][pivot:])
            else:
                child = list(gene_p[1][:pivot] + gene_p[0][pivot:])
        else:
            "DEU PAU!!!!!"
            exit()

        # print("Pre-mutation:\t", ''.join(child))

        # Mutation
        for i in range(len(child)):
            if np.random.rand() <= mutax:
                if child[i] == '0':
                    child[i] = '1'
                else:
                    child[i] = '0'

        # print("Post-mutation:\t", ''.join(child))
        
        # Convert original child from binary to int
        child = [-1, 0, 0, (int('0b' + ''.join(child[:bits]), 2), int('0b' + ''.join(child[bits:]), 2)),  99999, 99999]

        # Evaluate children fitness
        child[1], child[2] = fitness(child[3], bits, xmin, xmax, ymin, ymax)

        """
        if local_search:
            # Random generate slots to perform local change
            houses = np.random.choice(len(child), 0.1*len(child))
            # Perform local changes on original child
            for pos in houses:
                neighbour = copy.copy(child)
                if child[pos] == '0':
                    neighbour[pos] = '1'
                else:
                    neighbour[pos] = '0'
                
                # Convert modified child from binary to int and save it into list of children
                children.append([0, (int('0b' + ''.join(neighbour[:bits]), 2), int('0b' + ''.join(neighbour[bits:]), 2))])
        """

        # Add child to last position of population
        specimens.append(child)

    # Reset all tiers and sort population
    for spec in specimens:
        spec[0] = -1

    specimens.sort(reverse=True)

    aux_specimens = []
    tier = 0

    # Ranking process
    while specimens:

        f2 = np.array([x[2] for x in specimens])
        for i in range(len(specimens)):
            G = (f2 > f2[i])
            if np.all(G[:i]):
                specimens[i][0] = tier
                aux_specimens.append(specimens[i])

        tier += 1

        for i in range(len(specimens) - 1, -1, -1):
            if specimens[i] in aux_specimens:
                specimens.remove(specimens[i])

    # Copy specimens ranked in tiers and organized
    specimens = aux_specimens

    for spec in specimens:
        spec[4], spec[5] = 99999, 99999

    # Calculate normalized crowding distance for each tier
    for tiers in range(specimens[-1][0] + 1):
        aux = [x for x in specimens if x[0] == tiers]
        for i in range(1, len(aux) - 1):
            d = (np.sqrt(((aux[i][1] - aux[i - 1][1]) / (aux[0][1] - aux[-1][1])) ** 2 +
                         ((aux[i][2] - aux[i - 1][2]) / (aux[0][2] - aux[-1][2])) ** 2),
                 np.sqrt(((aux[i][1] - aux[i + 1][1]) / (aux[0][1] - aux[-1][1])) ** 2 +
                         ((aux[i][2] - aux[i + 1][2]) / (aux[0][2] - aux[-1][2])) ** 2))
            aux[i][4], aux[i][5] = min(d), max(d)

    for spec in specimens:
        print(spec)

    # Add counter of generations
    gen += 1

    if print_gen:
        print("Generation #%d" % gen)
        print(30*"-")
        for spec in specimens:
            if spec[1] == child:
                print("\x1b[;32;49m%d: %.4f, %.4f \x1b[0m" % (specimens.index(spec) + 1, spec[1][0]/(2.**bits)*(xmax-xmin) + xmin, spec[1][1]/(2.**bits)*(ymax-ymin) + ymin))
            else:
                print("%d: %.4f, %.4f" % (specimens.index(spec) + 1, spec[1][0]/(2.**bits)*(xmax-xmin) + xmin, spec[1][1]/(2.**bits)*(ymax-ymin) + ymin))
        print("\n\n")

# print("Best value: %.4f, %.4f" % (specimens[0][1][0]/(2.**bits)*(xmax - xmin) + xmin, specimens[0][1][1]/(2.**bits)*(ymax - ymin) + ymin))
# print("Objective function: %.6f" % specimens[0][0])

# Plot f1 versus f2 of every individual
plt.scatter([x[1] for x in specimens], [x[2] for x in specimens])
plt.show()
