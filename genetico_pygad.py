import random
import numpy as np
from deap import base, creator, tools, algorithms
import pygad
import time

n = 10  
p = 3   

d = np.random.randint(1, 100, size=(n, n))
np.fill_diagonal(d, 0) 

def evaluate(individual):
    medians = [i for i in range(n) if individual[i] == 1]
    if not medians:
        return (float('inf'),)  
    total_distance = 0
    for i in range(n):
        if individual[i] == 0: 
            total_distance += min(d[i][j] for j in medians)
    return (total_distance,)


def fitness_func(ga_instance, solution, solution_idx):
    medians = [i for i in range(n) if solution[i] == 1]
    if not medians:
        return float('-inf')  
    total_distance = 0
    for i in range(n):
        if solution[i] == 0:  
            total_distance += min(d[i][j] for j in medians)
    return -total_distance 

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual():
    ind = [0] * n
    selected = random.sample(range(n), p)
    for idx in selected:
        ind[idx] = 1
    return ind

toolbox = base.Toolbox()
toolbox.register(
    "individual", 
    tools.initIterate, 
    creator.Individual, 
    create_individual
)
toolbox.register(
    "population", 
    tools.initRepeat, 
    list, 
    toolbox.individual
)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament, tournsize=3)

def check_validity(ind):
    ones = sum(ind)
    if ones != p:
        if ones > p:
            indices = [i for i, val in enumerate(ind) if val == 1]
            for _ in range(ones - p):
                ind[random.choice(indices)] = 0
        else:
            indices = [i for i, val in enumerate(ind) if val == 0]
            for _ in range(p - ones):
                ind[random.choice(indices)] = 1
    return ind

def mutate_and_validate(ind, indpb):
    tools.mutShuffleIndexes(ind, indpb)
    return check_validity(ind),

toolbox.register("mutate", mutate_and_validate, indpb=0.05)

def run_deap():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
                        stats=stats, halloffame=hof, verbose=False)

    return hof[0], evaluate(hof[0])[0]

def run_pygad():
    def create_solution():
        solution = [0] * n
        selected = random.sample(range(n), p)
        for idx in selected:
            solution[idx] = 1
        return solution

    initial_population = [create_solution() for _ in range(300)]

    ga_instance = pygad.GA(
        num_generations=40,
        num_parents_mating=2,
        fitness_func=fitness_func,
        sol_per_pop=300,
        num_genes=n,
        initial_population=initial_population,
        gene_type=int,
        mutation_type="random",
        mutation_percent_genes=10,  
        on_generation=None
    )
    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    return solution, -solution_fitness

def main():
    start_time_deap = time.time()
    best_solution_deap, best_fitness_deap = run_deap()
    end_time_deap = time.time()

    start_time_pygad = time.time()
    best_solution_pygad, best_fitness_pygad = run_pygad()
    end_time_pygad = time.time()


    print("\nPygad:")
    print("Melhor solução encontrada:", best_solution_pygad)
    print("Distância total:", best_fitness_pygad)
    print("Tempo de execução: {:.2f} segundos".format(end_time_pygad - start_time_pygad))

    def print_matrix_with_medians(medians, distances):
        print("\nMatriz de distâncias com medianas (1 significa mediana):")
        for i in range(n):
            row = ""
            for j in range(n):
                if i in medians and j in medians:
                    row += "  M  "
                else:
                    row += f"{distances[i][j]:4d} "
            print(row)

    print("\nMatriz de distâncias - Pygad:")
    print_matrix_with_medians([i for i in range(n) if best_solution_pygad[i] == 1], d)

if __name__ == "__main__":
    main()
