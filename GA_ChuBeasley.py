import numpy as np
import copy


# Function that evaluates fitness of a given individual
def fitness(crom, n, bound):

    import numpy as np

    x = (crom/(2.**n))*(bound[1] - bound[0]) + bound[0]
    y = 2 - x

    return x*np.sin(10*np.pi*x) + y*np.cos(3*np.pi*(y**2))


# Flags
print_gen = not False
local_search = True

# Limits of x
xmin = -1
xmax = 2

# Limits of y
ymin = -.5
ymax = 1.8

# Boundaries for x
boundaries = (max(xmin, 2 - ymax), min(xmax, 2 - ymin))

# Size of population
pop_size = 100

# Maximum number of generations
max_generation = 1000

# Maximum number of consecutive generations with no change in population
max_unchange = max(3, int(.02*max_generation))

# Precision
digits = 6

# Mutation tax
mutax = 0.1

# Crossover tax
crosstax = 0.8

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

    # This loop prevents creating repeated individuals
    while [0, a] in specimens:
        a = np.random.randint(2**bits)

    # Add new individual to population
    specimens.append([0, a])

# Fitness calculation
for spec in specimens:
    spec[0] = fitness(spec[1], bits, boundaries)

# Sorting specimens for best to worst
specimens.sort(reverse=True)

if print_gen:
    print "Initial Generation"
    print 30*"-"
    for spec in specimens:
        print "%d: %.4f, %.4f" % (specimens.index(spec) + 1, spec[1]/(2.**bits)*(boundaries[1] - boundaries[0]) + boundaries[0], 2 - (spec[1]/(2.**bits)*(boundaries[1] - boundaries[0]) + boundaries[0]))
    print "\n\n"

while gen < max_generation and unchange < max_unchange:

    # Initiate children as one specimen that already exists, forcing code to enter the following 'while' loop
    children = [[0, specimens[0][1]]]

    # This loop prevents parents to generate a child that already exists on the population
    while children[0][1] in zip(*specimens)[1]:

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
        
        gene_p = [bin(parent[0])[2:].zfill(bits), bin(parent[1])[2:].zfill(bits)]
        
        # Crossover to generate original child
        if len(gene_p[0]) == len(gene_p[1]):
            
            # Set pivot point. If pivot point is zero, one of the parents will be passed on
            if np.random.rand() <= crosstax:
                pivot = np.random.randint(1, len(gene_p[0]) - 1)
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

        # print "Pre-mutation:\t", ''.join(child)

        # Mutation
        for i in range(len(child)):
            if np.random.rand() <= mutax:
                if child[i] == '0':
                    child[i] = '1'
                else:
                    child[i] = '0'

        # print "Post-mutation:\t", ''.join(child)
        
        # Convert original child from binary to int
        children = [[0, int('0b' + ''.join(child[:bits]), 2)]]
        
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
                children.append([0, int('0b' + ''.join(neighbour[:bits]), 2)])
        
        # Calculate fitness function for every children
        for kid in children:
            kid[0] = fitness(kid[1], bits, boundaries)
        
        # Sort children
        children.sort(reverse=True)

    # Add best children to last position of population
    specimens.append(children[0])
    
    # If child is better than worst individual, add it to population
    if specimens[-1][0] >= specimens[-2][0]:
        specimens = sorted(specimens, reverse=True)[:pop_size]
        unchange = 0
    else:
        specimens = specimens[:pop_size]
        unchange += 1

    # Add counter of generations
    gen += 1

    if print_gen:
        print "Generation #%d" % gen
        print 30*"-"
        for spec in specimens:
            if spec[1] == child:
                print "\x1b[;32;49m%d: %.4f, %.4f \x1b[0m" % (specimens.index(spec) + 1, spec[1]/(2.**bits)*(boundaries[1] - boundaries[0]) + boundaries[0], 2 - (spec[1]/(2.**bits)*(boundaries[1] - boundaries[0]) + boundaries[0]))
            else:
                print "%d: %.4f, %.4f" % (specimens.index(spec) + 1, spec[1]/(2.**bits)*(boundaries[1] - boundaries[0]) + boundaries[0], 2 - (spec[1]/(2.**bits)*(boundaries[1] - boundaries[0]) + boundaries[0]))
        print "\n\n"

xbest = specimens[0][1]/(2.**bits)*(boundaries[1] - boundaries[0]) + boundaries[0]
ybest = 2 - xbest

print "Best value: %.4f, %.4f" % (xbest, ybest)
print "Objective function: %.6f" % specimens[0][0]
print gen, unchange
