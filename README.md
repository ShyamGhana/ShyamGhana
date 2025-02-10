import heapq

def a_star_search(graph, start, goal, heuristic):
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic[start]
    
    came_from = {}
    
    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor, cost in graph[current].items():
            temp_g_score = g_score[current] + cost

            if temp_g_score < g_score[neighbor]:
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic[neighbor]
                heapq.heappush(open_list, (f_score[neighbor], neighbor))
                came_from[neighbor] = current

    return None
                
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'D': 2, 'E': 5},
    'C': {'A': 4, 'F': 1},
    'D': {'B': 2, 'G': 2},
    'E': {'B': 5},
    'F': {'C': 1, 'G': 3},
    'G': {'D': 2, 'F': 3},
}

heuristic = {
    'A': 6, 'B': 4, 'C': 4, 'D': 3,
    'E': 2, 'F': 2, 'G': 0
}

start = 'A'
goal = 'G'

path = a_star_search(graph, start, goal, heuristic)

if path:
    print("Path found:", path)
else:
    print("No path found.")

