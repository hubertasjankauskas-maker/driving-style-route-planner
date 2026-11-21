import networkx as nx

PROFILE_WEIGHTS = {
    "Standartinė navigacija": (0.0, 0.0, 0.0),
    "Atsargus / Pradedantysis": (1.5, 0.8, 0.2),
    "Normalus": (0.2, 0.2, 0.2),
    "Agresyvus": (0.1, 1.2, 2.0),
}

def build_city_graph():
    G = nx.DiGraph()
    edges = [
        ("A", "B", 3.0, 0.7, 0.2, 0.4),
        ("B", "E", 3.0, 0.8, 0.2, 0.5),
        ("E", "H", 2.5, 0.6, 0.1, 0.5),
        ("H", "L", 3.0, 0.3, 0.1, 0.2),
        ("A", "C", 3.5, 0.1, 0.1, 0.2),
        ("C", "F", 4.0, 0.2, 0.1, 0.3),
        ("F", "I", 3.5, 0.2, 0.1, 0.4),
        ("I", "L", 3.0, 0.15, 0.1, 0.2),
        ("A", "D", 3.2, 0.4, 0.2, 0.05),
        ("D", "G", 3.3, 0.5, 0.1, 0.05),
        ("G", "K", 3.0, 0.4, 0.1, 0.10),
        ("K", "L", 3.0, 0.2, 0.1, 0.10),
    ]
    for u, v, t, c, k, p in edges:
        G.add_edge(u, v, time=t, complexity=c, curvature=k, pedestrian=p)
    return G

def plan_route(G, start, end, profile_name):
    alpha, beta, gamma = PROFILE_WEIGHTS[profile_name]

    def edge_cost(u, v, d):
        return d["time"] * (1.0 + alpha * d["complexity"] + beta * d["curvature"] + gamma * d["pedestrian"])

    path = nx.dijkstra_path(G, start, end, weight=edge_cost)
    total_time = sum(G[u][v]["time"] for u, v in zip(path[:-1], path[1:]))
    total_comp = sum(G[u][v]["complexity"] for u, v in zip(path[:-1], path[1:]))
    total_ped = sum(G[u][v]["pedestrian"] for u, v in zip(path[:-1], path[1:]))

    return path, total_time, total_comp, total_ped

if __name__ == "__main__":
    city_graph = build_city_graph()
    print("=== MARŠRUTO PLANAVIMAS PAGAL VAIRAVIMO STILIŲ (A -> L) ===")
    for profile in PROFILE_WEIGHTS:
        path, t, comp, ped = plan_route(city_graph, "A", "L", profile)
        print(f"{profile:26s} | Maršrutas: {' -> '.join(path)} | Laikas: {t:.1f} min. | "
              f"Sankryžų sudėt.: {comp:.2f} | Pėsčiųjų zonos: {ped:.2f}")
