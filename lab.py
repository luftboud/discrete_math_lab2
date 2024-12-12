"""
Lab 2
"""
import math
import random
from collections import defaultdict


def read_incidence_matrix(filename: str) -> list[list]:
    """
    :param str filename: path to file
    :returns list[list]: the incidence matrix of a given graph
    >>> read_incidence_matrix('input.dot')
    [[1, 1, -1, 0, -1, 0], \
[-1, 0, 1, 1, 0, -1], \
[0, -1, 0, -1, 1, 1]]
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.readlines()[1:-1]
    content = [el.strip()[:-1] for el in content]
    vertices = set()
    for el in content:
        vertices.add(int(el[0]))
    matrix = [[0 for _ in range(len(content))] for _ in range(len(vertices))]
    for i, el in enumerate(content):
        matrix[int(el[0])][i] = 1
        matrix[int(el[-1])][i] = -1
    return matrix


def read_adjacency_matrix(filename: str) -> list[list]:
    """
    :param str filename: path to file
    :returns list[list]: the adjacency matrix of a given graph
    >>> read_adjacency_matrix('input.dot')
    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.readlines()[1:-1]
    content = [el.strip()[:-1] for el in content]
    content_dict = {}
    for el in content:
        content_dict.setdefault(int(el[0]),[])
        content_dict[int(el[0])].append(int(el[-1]))
    matrix = [[0 for _ in range(len(content_dict))] for _ in range(len(content_dict))]
    for vert1, vertices in content_dict.items():
        for vert2 in vertices:
            matrix[vert1][vert2] = 1
    return matrix


def read_adjacency_dict(filename: str) -> dict[int, list[int]]:
    """
    :param str filename: path to file
    :returns dict: the adjacency dict of a given graph
    >>> read_adjacency_dict('input.dot')
    {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.readlines()[1:-1]
    content = [el.strip()[:-1] for el in content]
    content_dict = {}
    for el in content:
        content_dict.setdefault(int(el[0]),[])
        content_dict[int(el[0])].append(int(el[-1]))
    return content_dict


def iterative_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param list[list] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    checked = set()
    trail = []
    queue = [start]
    while queue:
        vert = queue.pop(0)
        if vert not in checked:
            trail.append(vert)
            checked.add(vert)
            queue.extend(graph[vert])
    return trail




def iterative_adjacency_matrix_dfs(graph: list[list], start: int) ->list[int]:
    """
    :param dict graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    checked = set()
    trail = []
    queue = [start]
    while queue:
        vert = queue.pop(0)
        if vert not in checked:
            trail.append(vert)
            checked.add(vert)
            queue.extend([i for i, el in enumerate(graph[vert]) if el == 1])
    return trail



# def recursive_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
#     """
#     :param list[list] graph: the adjacency list of a given graph
#     :param int start: start vertex of search
#     :returns list[int]: the dfs traversal of the graph
#     >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
#     [0, 1, 2]
#     >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
#     [0, 1, 2, 3]
#     """
#     pass


# def recursive_adjacency_matrix_dfs(graph: list[list[int]], start: int) ->list[int]:
#     """
#     :param dict graph: the adjacency matrix of a given graph
#     :param int start: start vertex of search
#     :returns list[int]: the dfs traversal of the graph
#     >>> recursive_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
#     [0, 1, 2]
#     >>> recursive_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
#     [0, 1, 2, 3]
#     """
#     pass


# def iterative_adjacency_dict_bfs(graph: dict[int, list[int]], start: int) -> list[int]:
#     """
#     :param list[list] graph: the adjacency list of a given graph
#     :param int start: start vertex of search
#     :returns list[int]: the bfs traversal of the graph
#     >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
#     [0, 1, 2]
#     >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
#     [0, 1, 2, 3]
#     """
#     pass


# def iterative_adjacency_matrix_bfs(graph: list[list[int]], start: int) ->list[int]:
#     """
#     :param dict graph: the adjacency matrix of a given graph
#     :param int start: start vertex of search
#     :returns list[int]: the bfs traversal of the graph
#     >>> iterative_adjacency_matrix_bfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
#     [0, 1, 2]
#     >>> iterative_adjacency_matrix_bfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
#     [0, 1, 2, 3]
#     """
#     pass



def floyd_warshall(graph: list[list[int]]) -> list[list[int]]:
    """
    Computes the shortest paths between all pairs of vertices
    using the Floyd-Warshall algorithm.
    Iterates over all nodes incrementing distances, compares direct distances
    and distances across intermediade nodes `k` and chooces the smallest
    :param graph: the graph in an adjacency matrix form
    :return: 2D-matrix of shortest distances
    >>> floyd_warshall([[0, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]])
    [[0, 2, 2, 1], [1, 0, 1, 2], [1, 1, 0, 2], [1, 1, 1, 0]]
    """
    n = len(graph)

    distances = [[0] * n for _ in range(n)]
    for i in range(n):
        distances[i][i] = 0
        for j in range(n):
            distances[i][j] = float('inf') if not graph[i][j] and i != j else graph[i][j]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])

    return distances


def adjacency_matrix_radius(graph: list[list]) -> int:
    """
    Utilizes Floyd-Warshall algorithm to find eccentricities for the given matrix.
    Picks the smallest eccentricity value which is the radius
    :param graph: the adjacency matrix of a given graph
    :returns int: the r of the graph
    >>> adjacency_matrix_radius([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    1
    >>> adjacency_matrix_radius([[1, 0, 0], [0, 0, 1], [1, 0, 0]])
    2
    >>> adjacency_matrix_radius([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    3
    >>> adjacency_matrix_radius([[0, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    inf
    """
    if not (graph and len(graph) == len(graph[0])):
        return 0

    paths = floyd_warshall(graph)
    eccentricities = [max(row) for row in paths]
    r = min(eccentricities)

    return r


# def adjacency_dict_radius(graph: dict[int: list[int]]) -> int:
#     """
#     :param dict graph: the adjacency list of a given graph
#     :returns int: the radius of the graph
#     >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1]})
#     1
#     >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [1]})
#     2
#     """
#     pass


def generate_adjacency_graph(
        vertices: int,
        density: float = .5,
        type_='matrix'
) -> list[list[int]] | dict[int, list[int]] | None:
    """
    Gets a number of vertices a density index and generates a graph
    in an asked representation type.
    The density index is a real number in [0,1], helps to calculate number of edges
    and distribute them during generation.
    The density index is normalized if not in [0,1]
    :param vertices: number of vertices in the graph
    :param density: float density index to determine level of connectivity between vertices
    :param type_: only `matrix` or `dict` allowed
    :return: either matrix or dict representation of the generated graph, if type incorrect - None
    """
    if not (0 <= density <= 1):
        scale = math.ceil(math.log10(density))
        normalized = density / 10 ** scale
        density = round(normalized, 3)

    precision = len(str(density).partition('.')[-1])
    conn_chance, disconn_chance = density * 10 ** precision, (1 - density) * 10 ** precision
    connection_probability = (1, ) * round(conn_chance) + (0, ) * round(disconn_chance)

    match type_:
        case 'matrix': return generate_adjacency_matrix(vertices, connection_probability)
        case 'dict': return generate_adjacency_dict(vertices, connection_probability)
        case _: return None


def generate_adjacency_matrix(n: int, connection_chance: tuple[int]) -> list[list[int]]:
    """
    Generates a 2D boolean adjacency matrix
    :param n: number of vertices
    :param connection_chance: boolean tuple, represents a connectivity chance/percentage between nodes
    :return: the matrix
    """
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = random.choice(connection_chance)

    return matrix


def generate_adjacency_dict(n: int, connection_chance: tuple[int]) -> dict[int, list[int]]:
    """
    Generates an adjacency dict in a format {node: list of adjacent nodes}
    :param n: number of vertices
    :param connection_chance: boolean tuple, represents a connectivity chance/percentage between nodes
    :return: the dictionary
    """
    graph = defaultdict(list)

    for i in range(n):
        for j in range(n):
            if i != j and random.choice(connection_chance):
                graph[i].append(j)

    return graph


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()

    import doctest
    print(doctest.testmod())
    # from pprint import pprint
    # m = generate_adjacency_graph(50, density=0.01, type_='matrix')
    # pprint(m, width=300)
    # print(adjacency_matrix_radius(m))
