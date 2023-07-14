import numpy as np
import time
from deap import base, creator, tools, algorithms

from task_rnd_triang_with_interrupts_stdev_new_R2_deterministic import *
from functions_for_simheuristic_v12 import *

#array to store all found solutions
solutions = []

nrcandidates = 20
timestamps = []

# calculate the costs for each project by utilizing the corresponding functions inside task_rnd_triang_with_interrupts_stdev_new_R2_deterministic.py
# and store all of them in an array called bdgtperproject_matrix

#I define a candidate array of size nr candidates with all integer values: ones
candidatearray = np.ones(nrcandidates)

#I define an initial array of indexes with all candidates ranging from 0 to nrcandidates-1
initial_projection_indexes = np.arange(nrcandidates)

#first simulation to get all cdfs for cost & benefits before optimization step (may_update: was 1000)
det_results1 = calc_det(candidatearray, 1)

# det_results1[0] corresponds to the project costs and det_results1[1] to the project benefits (NPV)

# write the first timestamp and label to the list
timestamps.append(('First deterministic point estimate of budgets and NPV for each project', time.time()))

# extract first column of the matrix to get the budgeted costs of each project and store it in bdgtperproject_matrix
bdgtperproject_matrix = det_results1[0]
print("bdgtperproject_matrix: ", bdgtperproject_matrix)
# extract second column of the matrix to get the NPV of each project and store it in npvperproject_matrix
npvperproject_matrix = det_results1[1]
print("npvperproject_matrix: ", npvperproject_matrix)
# define the budget constraint
maxbdgt = 8000

# copy the array with all MCS results  
det_df0 = pd.DataFrame(data=det_results1[0]).T  
det_col_names = ["P{:02d}".format(i+1) for i in range(nrcandidates)]  
det_df0.rename(columns=dict(enumerate(det_col_names)), inplace=True)  


# *************************  Start of optimization step *************************

# write the second timestamp (substract the current time minus the previously stored timestamp) and label to the list
timestamps.append(('Second MCS with correlated cost and NPV for each project', time.time()))

# Defining the fitness function
def evaluate(individual, bdgtperproject, npvperproject, maxbdgt):
    total_cost = 0
    total_npv = 0
    #multiply dataframe 10r by the chosen portfolio to reflect the effect of the projects that are chosen
    pf_df10r = det_df0 * individual
    #sum the rows of the new dataframe to calculate the total cost of the portfolio
    pf_cost10r = pf_df10r.sum(axis=1)
    #extract the maximum of the resulting costs
    #maxcost10r = max(pf_cost10r)
    #print("max cost:")
    #print(maxcost10r)
    #count how many results were higher than maxbdgt
    #count = 0
    #for i in range(pf_cost10r.__len__()):
    #    if pf_cost10r[i] > maxbdgt:
    #        count = count + 1
    #array storing the portfolio risk not to exceed 3.800 Mio.€, as per-one risk units
    #portfolio_confidence = 1-count/iterations
    #print("portfolio confidence:")
    #print(portfolio_confidence)
    bdgtperproject = np.squeeze(bdgtperproject)
    bdgtperproject = [int(x) for x in bdgtperproject]
    npvperproject = np.squeeze(npvperproject)
    npvperproject = [int(x) for x in npvperproject]
    for i in range(nrcandidates):
        #print(total_cost)
        if individual[i] == 1:
            total_cost += bdgtperproject[i]
            #total_cost += PROJECTS[i][0]
            # add the net present value of the project to the total net present value of the portfolio
            total_npv += npvperproject[i]
            #total_npv += npv[i][1]

    if total_cost > maxbdgt:
        return 0,
    return total_npv,

# Define the genetic algorithm parameters
POPULATION_SIZE = 180 #was 100 #was 50
P_CROSSOVER = 0.4
P_MUTATION = 0.6
MAX_GENERATIONS = 300 #was 500 #was 200 #was 100
HALL_OF_FAME_SIZE = 3

# Create the individual and population classes based on the list of attributes and the fitness function # was weights=(1.0,) returning only one var at fitness function
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #was "(100000.0, 1.0)" instead of "(1.0)"
# create the Individual class based on list
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the toolbox
toolbox = base.Toolbox()
# register a function to generate random integers (0 or 1) for each attribute/gene in an individual
toolbox.register("attr_bool", random.randint, 0, 1) #was "0, 1" instead of "1"
# register a function to generate individuals (which are lists of several -nrcandidates- 0s and 1s -genes-
# that represent the projects to be included in the portfolio)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, nrcandidates)
# register a function to generate a population (a list of individuals -candidate portfolios-)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# register the goal / fitness function
toolbox.register("evaluate", evaluate, bdgtperproject=bdgtperproject_matrix, npvperproject=npvperproject_matrix, maxbdgt=maxbdgt)
# register the crossover operator (cxTwoPoint) with a probability of 0.9 (defined above)
toolbox.register("mate", tools.cxTwoPoint)
# register a mutation operator with a probability to flip each attribute/gene of 0.05.
# indpb is the independent probability for each gene to be flipped and P_MUTATION is the probability of mutating an individual
# The difference between P_MUTATION and indpb is that P_MUTATION determines whether an individual will be mutated or not,
# while indpb determines how much an individual will be mutated if it is selected for mutation.
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the hall of fame
hall_of_fame = tools.HallOfFame(HALL_OF_FAME_SIZE)

# Define the statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", max)

# defining the function that maximizes the net present value of a portfolio of projects, while respecting the budget constraint (using a genetic algorithm)
def maximize_npv():
    # Empty the hall of fame
    hall_of_fame.clear()
    # Initialize the population
    population = toolbox.population(n=POPULATION_SIZE)
    for generation in range(MAX_GENERATIONS):
        # Vary the population
        offspring = algorithms.varAnd(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION)
        # Evaluate the new individuals fitnesses
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        # Update the hall of fame with the generated individuals
        hall_of_fame.update(offspring)
        # reorder the hall of fame so that the highest fitness individual is first
        # hall_of_fame.sort(key=itemgetter(0), reverse=True)
        population = toolbox.select(offspring, k=len(population))    
        record = stats.compile(population)
        # for each generation, print the max net present value of the portfolio and the total budget
        print(f"Generation {generation}: Max NPV = {record['max']}")

    #de momento me dejo de complicarme con el hall of fame y me quedo con el último individuo de la última generación
    # return the optimal portfolio from the hall of fame, their fitness and the total budget
    # print(hall_of_fame)
    #return hall_of_fame
    print("Hall of Fame:")
    print(hall_of_fame)
    print("Hall of Fame fitness:")
    print(hall_of_fame[0].fitness.values)


    #for i in range(HALL_OF_FAME_SIZE):
    #    print(hall_of_fame[i], hall_of_fame[i].fitness.values[0], hall_of_fame[i].fitness.values[1], portfolio_totalbudget(hall_of_fame[i], bdgtperproject_matrix))
    #print(hall_of_fame[0], hall_of_fame[0].fitness.values[0], hall_of_fame[0].fitness.values[1], portfolio_totalbudget(hall_of_fame[0], bdgtperproject_matrix))
    #print(hall_of_fame[1], hall_of_fame[1].fitness.values[0], hall_of_fame[1].fitness.values[1], portfolio_totalbudget(hall_of_fame[1], bdgtperproject_matrix))
    #print(hall_of_fame[2], hall_of_fame[2].fitness.values[0], hall_of_fame[2].fitness.values[1], portfolio_totalbudget(hall_of_fame[2], bdgtperproject_matrix))
    #return hall_of_fame[0], hall_of_fame[0].fitness.values[0][0], portfolio_totalbudget(hall_of_fame[0], bdgtperproject_matrix)
    return hall_of_fame

# execute the maximizer function to obtain the portfolio, and its npv and bdgt
projectselection = maximize_npv()

optimal_det_pf = det_df0 * projectselection[0]
cost_of_det_optimal_pf = optimal_det_pf.sum(axis=1).sum(axis=0)

print("Cost of deterministic optimal portfolio:")
print(cost_of_det_optimal_pf)

# assign the result from projectselection to the variable solutions
solutions.append(projectselection)