# Return 2D adjacency matrix of graph
def construct_graph(file):
    with open(file, 'r') as file:
        lines = file.readlines()
    
    graph = []
    for line in lines:
        if line == '\n':
            continue
        float_line = [float(x) for x in line.split()]
        graph.append(float_line)

    return graph