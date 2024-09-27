import utils
import random
from collections import defaultdict
from functools import cache


@cache
def path_cost(path):
    global graph
    
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]
    return cost


def update_pheromones(pheromones, paths, rho):
    global graph

    pheromone_delta = defaultdict(int)
    for path in paths:
        cost = path_cost(tuple(path))
        for i in range(len(path) - 1):
            pheromone_delta[(path[i], path[i + 1])] += 1 / cost
    
    for i in range(len(pheromones)):
        for j in range(len(pheromones)):
            pheromones[i][j] = (1 - rho) * pheromones[i][j] + pheromone_delta[(i, j)]


def pick_next(pheromones, visited, current, alpha, beta):
    global graph

    unvisited = [i for i in range(len(graph)) if not visited[i]]
    probs = [0] * len(graph)
    
    for i in unvisited:
        probs[i] = (pheromones[current][i] ** alpha) / (graph[current][i] ** beta)
    
    total = sum(probs)
    probs = [p / total for p in probs]
    
    r = random.random()
    for i in range(len(probs)):
        r -= probs[i]
        if r <= 0:
            return i
    
    return unvisited[-1]


def construct_path(pheromones, alpha, beta):
    global graph

    path = [random.randint(0, len(graph) - 1)]
    visited = [False] * len(graph)
    visited[path[0]] = True
    
    for _ in range(len(graph) - 1):
        current = path[-1]
        next_city = pick_next(pheromones, visited, current, alpha, beta)
        path.append(next_city)
        visited[next_city] = True

    path.append(path[0])

    return path


def tsp(ants, alpha, beta, rho, limit=100):
    global graph

    pheromones = [[random.random() for _ in range(len(graph))] for _ in range(len(graph))]
    overall_best_path = []
    overall_best_cost = float('inf')
    
    for t in range(limit):
        paths = []
        curr_best_path = []
        curr_best_cost = float('inf')

        for _ in range(ants):
            path = construct_path(pheromones, alpha, beta)
            paths.append(path)
            if path_cost(tuple(path)) < curr_best_cost:
                curr_best_cost = path_cost(tuple(path))
                curr_best_path = path

        if curr_best_cost < overall_best_cost:
            overall_best_cost = curr_best_cost
            overall_best_path = curr_best_path

        update_pheromones(pheromones, paths, rho)
        
        print('Iteration', t + 1)
        print('Best Path in Iteration:', curr_best_path)
        print('Best Cost in Iteration:', curr_best_cost)
        print('Best Cost Overall:', overall_best_cost)
        print()
        

if __name__ == '__main__':
    graph = utils.construct_graph('data/2085.txt')
    
    ants = 100
    alpha = 1
    beta = 3
    rho = 0.5
    tsp(ants, alpha, beta, rho, limit=500)