import numpy as np
from pyswarm import pso

n = 20  
p = 5   

d = np.random.randint(1, 100, size=(n, n))
np.fill_diagonal(d, 0) 

def fitness_func(solution):
    medians = np.argsort(solution)[:p] 
    total_distance = 0
    for i in range(n):
        if i not in medians:
            total_distance += min(d[i][j] for j in medians)
    return total_distance

def lower_bound(dim):
    return [0] * dim

def upper_bound(dim):
    return [(n-1)] * dim

dim = n

lb = lower_bound(dim)
ub = upper_bound(dim)

def main():
    import time
    start_time = time.time()
    best_pos, best_cost = pso(fitness_func, lb, ub, swarmsize=100, maxiter=100)
    end_time = time.time()

  
    best_medians = np.argsort(best_pos)[:p]

    print("Melhor solução encontrada (índices das medianas):", best_medians)
    print("Distância total:", best_cost)
    print("Tempo de execução: {:.2f} segundos".format(end_time - start_time))

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

    print("\nMatriz de distâncias - PSO:")
    print_matrix_with_medians(best_medians, d)

if __name__ == "__main__":
    main()
