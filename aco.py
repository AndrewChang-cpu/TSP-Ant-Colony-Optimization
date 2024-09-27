import utils


def tsp(graph, ants, alpha, beta, rho):
    pass


if __name__ == '__main__':
    graph = utils.construct_graph('data/5.txt')
    
    ants = 10
    alpha = 1
    beta = 2
    rho = 0.1
    tsp(graph, ants, alpha, beta, rho)