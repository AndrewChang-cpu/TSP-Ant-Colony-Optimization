import utils
import random


def path_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]
    
    cost += graph[path[-1]][path[0]]
    return cost


def update_pheromones(pheromones, path, cost, rho):
    for i in range(len(path) - 1):
        pheromones[path[i]][path[i + 1]] = (1 - rho) * pheromones[path[i]][path[i + 1]] + 1 / cost
        pheromones[path[i + 1]][path[i]] = (1 - rho) * pheromones[path[i + 1]][path[i]] + 1 / cost
    
    pheromones[path[-1]][path[0]] = (1 - rho) * pheromones[path[-1]][path[0]] + 1 / cost
    pheromones[path[0]][path[-1]] = (1 - rho) * pheromones[path[0]][path[-1]] + 1 / cost


def pick_next(graph, pheromones, visited, current, alpha, beta):
    unvisited = [i for i in range(len(graph)) if not visited[i]]
    probs = [0 for _ in range(len(graph))]
    
    for i in unvisited:
        probs[i] = (pheromones[current][i] ** alpha) * ((1 / graph[current][i]) ** beta)
    
    total = sum(probs)
    probs = [p / total for p in probs]
    
    r = random.random()
    for i in range(len(probs)):
        r -= probs[i]
        if r <= 0:
            return i
    
    return unvisited[-1]


def tsp(graph, ants, alpha, beta, rho, limit=100):
    # TODO: store best path, run through logic again, am i supposed to update pheromonoes after each ant?
    pheromones = [[random.random() for _ in range(len(graph))] for _ in range(len(graph))]
    
    for t in range(limit):
        for _ in range(ants):
            path = [random.randint(0, len(graph) - 1)]
            visited = [False for _ in range(len(graph))]
            visited[path[0]] = True
            
            for _ in range(len(graph) - 1):
                current = path[-1]
                next = pick_next(graph, pheromones, visited, current, alpha, beta)
                path.append(next)
                visited[next] = True
            
            cost = path_cost(graph, path)
            update_pheromones(pheromones, path, cost, rho)
        
        print('Iteration', t + 1)
        print('Best path:', path)
        print('Cost:', cost)
        print()
        

if __name__ == '__main__':
    graph = utils.construct_graph('data/5.txt')
    
    ants = 10
    alpha = 1
    beta = 2
    rho = 0.1
    tsp(graph, ants, alpha, beta, rho)