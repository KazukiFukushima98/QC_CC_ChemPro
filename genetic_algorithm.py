import random
from deap import base, creator, tools
import numpy as np
import time
import math


def custom_eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen,
                               stats=None, halloffame=None, verbose=__debug__,
                               elite_size=10):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Initial evaluation
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    if halloffame is not None and len(halloffame) > 0:
        best = halloffame[0]
        if abs(best[0]) < 0.1 and (abs(best[1] - 3.7) < 0.1 or abs(best[1] - 3.76) < 0.01) and best[2] == 0 and best[
            3] == 1:
            return population, logbook

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluation
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Convergence check
        if halloffame is not None and len(halloffame) > 0:
            best = halloffame[0]
            if abs(best[0]) < 0.1 and (abs(best[1] - 3.7) < 0.1 or abs(best[1] - 3.76) < 0.01) and best[2] == 0 and \
                    best[3] == 1:
                break

    return population, logbook


xlb = 0
xub = 5
c = [1.5, 1, 6.1, 5.9]
alpha = [0.75, 0.8]
d = 3

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def create_individual():
    x0 = random.uniform(xlb, xub)
    x1 = random.uniform(xlb, xub)
    y0 = random.randint(0, 1)
    y1 = random.randint(0, 1)
    return creator.Individual([x0, x1, y0, y1])


toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    x0, x1, y0, y1 = individual
    obj = c[0] * x0 ** 2 + c[1] * x1 ** 2 + c[2] * y0 + c[3] * y1
    penalty = 0
    penalty_weight = 10000

    g1 = xlb * y0 - x0
    if g1 > 0:
        penalty += penalty_weight * g1 ** 2
    g2 = x0 - xub * y0
    if g2 > 0:
        penalty += penalty_weight * g2 ** 2
    g3 = xlb * y1 - x1
    if g3 > 0:
        penalty += penalty_weight * g3 ** 2
    g4 = x1 - xub * y1
    if g4 > 0:
        penalty += penalty_weight * g4 ** 2
    g5 = d - (alpha[0] * x0 + alpha[1] * x1)
    if g5 > 0:
        penalty += penalty_weight * g5 ** 2

    if x0 < xlb:
        penalty += penalty_weight * (xlb - x0) ** 2
    if x0 > xub:
        penalty += penalty_weight * (x0 - xub) ** 2
    if x1 < xlb:
        penalty += penalty_weight * (xlb - x1) ** 2
    if x1 > xub:
        penalty += penalty_weight * (x1 - xub) ** 2

    return obj + penalty,


toolbox.register("evaluate", evaluate)


def cxMixed(ind1, ind2):
    alpha_blend = 0.5
    for i in range(2):
        if random.random() < 0.5:
            gamma = (1. + 2. * alpha_blend) * random.random() - alpha_blend
            x1 = ind1[i]
            x2 = ind2[i]
            ind1[i] = (1. - gamma) * x1 + gamma * x2
            ind2[i] = gamma * x1 + (1. - gamma) * x2

    if random.random() < 0.5:
        ind1[2], ind2[2] = ind2[2], ind1[2]
    if random.random() < 0.5:
        ind1[3], ind2[3] = ind2[3], ind1[3]

    return ind1, ind2


def mutMixed(individual, indpb=0.2, sigma=2.0):
    for i in range(2):
        if random.random() < indpb:
            individual[i] += random.gauss(0, sigma)
            individual[i] = max(xlb, min(xub, individual[i]))

    for i in range(2, 4):
        if random.random() < indpb:
            individual[i] = 1 - individual[i]

    return individual,


toolbox.register("mate", cxMixed)
toolbox.register("mutate", mutMixed, indpb=0.2, sigma=2.0)
toolbox.register("select", tools.selTournament, tournsize=3)

count = 0
save_list = []

for i in range(100):
    GENERATIONS = 300
    POPULATION_SIZE = 300
    CROSSOVER_PROB = 0.7
    MUTATION_PROB = 0.2
    ELITE_SIZE = 0

    pop = toolbox.population(n=POPULATION_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(10)

    optimization_start_time = time.time()

    population, logbook = custom_eaSimpleWithElitism(
        pop, toolbox,
        cxpb=CROSSOVER_PROB,
        mutpb=MUTATION_PROB,
        ngen=GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    optimization_end_time = time.time()
    best = hof[0]

    is_success = (abs(best[0]) < 0.1 and
                  (abs(best[1] - 3.7) < 0.1 or abs(best[1] - 3.76) < 0.01) and
                  best[2] == 0 and best[3] == 1)

    actual_gens = logbook[-1]["gen"]
    elapsed = optimization_end_time - optimization_start_time

    save_list.append([best, best.fitness.values[0], elapsed, actual_gens, is_success])

success_trials = [s for s in save_list if s[4] == True]
failure_trials = [s for s in save_list if s[4] == False]

p_success = len(success_trials) / 100

all_times = [s[2] for s in save_list]
avg_time = np.mean(all_times)

success_times = [s[2] for s in success_trials]
success_gens = [s[3] for s in success_trials]

p_target = 0.99
tts = avg_time * math.log(1 - p_target) / math.log(1 - p_success)