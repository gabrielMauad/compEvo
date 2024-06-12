import random
import numpy as np
import time

n = 10 
p = 3  

d = np.random.randint(1, 100, size=(n, n))
np.fill_diagonal(d, 0)  

def evaluate(individual):
    medians = [i for i in range(n) if individual[i] == 1]
    if not medians:
        return float('inf')  
    total_distance = 0
    for i in range(n):
        if individual[i] == 0:  
            total_distance += min(d[i][j] for j in medians)
    return total_distance

def create_individual():
    ind = [0] * n
    selected = random.sample(range(n), p)
    for idx in selected:
        ind[idx] = 1
    return ind

def create_population(pop_size):
    return [create_individual() for _ in range(pop_size)]

def tournament_selection(pop, k=3):
    selected = random.sample(pop, k)
    selected.sort(key=evaluate)
    return selected[0]

def crossover(parent1, parent2):
    point = random.randint(1, n - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return fix_individual(child1), fix_individual(child2)

def mutate(individual, mutation_rate=0.01):
    for i in range(n):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return fix_individual(individual)

def fix_individual(individual):
    ones = sum(individual)
    if ones > p:
        indices = [i for i, val in enumerate(individual) if val == 1]
        for _ in range(ones - p):
            individual[random.choice(indices)] = 0
    elif ones < p:
        indices = [i for i, val in enumerate(individual) if val == 0]
        for _ in range(p - ones):
            individual[random.choice(indices)] = 1
    return individual

def genetic_algorithm(pop_size, generations, mutation_rate):
    pop = create_population(pop_size)
    for generation in range(generations):
        new_pop = []
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(pop)
            parent2 = tournament_selection(pop)
            child1, child2 = crossover(parent1, parent2)
            new_pop.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
        pop = new_pop
    best_individual = min(pop, key=evaluate)
    return best_individual, evaluate(best_individual)

def print_matrix_with_medians(medians, distances):
    print("\nMatriz de distâncias com medianas")
    for i in range(n):
        row = ""
        for j in range(n):
            if i in medians and j in medians:
                row += "  M  "
            else:
                row += f"{distances[i][j]:4d} "
        print(row)

start_time = time.time()
best_solution, best_fitness = genetic_algorithm(pop_size=100, generations=100, mutation_rate=0.01)
end_time = time.time()

print("Melhor solução encontrada:", best_solution)
print("Distância total:", best_fitness)
print("Tempo de execução: {:.2f} segundos".format(end_time - start_time))

print_matrix_with_medians([i for i in range(n) if best_solution[i] == 1], d)
