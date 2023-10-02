import networkx as nx
import matplotlib.pyplot as plt

# 定义边表数据
edges = [
    ("v0", "v1", 2), ("v1", "v2", 1), ("v2", "v3", 2), ("v3", "v4", 1), ("v4", "v5", 3),
    ("v5", "v6", 2), ("v6", "v7", 1), ("v7", "v8", 2), ("v8", "v9", 3), ("v9", "v10", 2),
    ("v0", "v11", 4), ("v11", "v5", 2), ("v11", "v12", 1), ("v12", "v13", 1), ("v13", "v9", 2),
    ("v6", "v14", 2), ("v14", "v15", 1), ("v15", "v16", 1), ("v16", "v17", 3), ("v17", "v10", 2)
]

# 查询数据
queries = [
    ("v0", "v10"),  # 查询1
    ("v0", "v9"),   # 查询2
    ("v0", "v17")   # 查询3
]

# 创建一个有向图并添加边
G = nx.DiGraph()
for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# 定义顶点的坐标，将布局更改为3x6的布局
pos = {}
spacing = 1.5
for i in range(3):
    for j in range(6):
        node = "v" + str(i*6 + j)
        if node in G.nodes:  # 确保只为我们关心的节点定义位置
            pos[node] = (j * spacing, -i * spacing)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制图形
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=600, edge_color="gray", ax=ax)

# 绘制边权重
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

# 绘制查询路径
query_colors = ["red", "green", "blue"]
offsets = [-0.2, 0, 0.2]
for index, (source, target) in enumerate(queries):
    path = nx.shortest_path(G, source, target, weight="weight")
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=query_colors[index], width=2, ax=ax, alpha=0.6, connectionstyle=f"arc3,rad={offsets[index]}")
    nx.draw_networkx_nodes(G, pos, nodelist=path[:-1], node_color="lightblue", node_size=600, alpha=0.6, ax=ax)

# 高亮度数最高的5个顶点
degree_sequence = sorted(G.degree, key=lambda x: x[1], reverse=True)
high_degree_nodes = [node[0] for node in degree_sequence[:5]]
nx.draw_networkx_nodes(G, pos, nodelist=high_degree_nodes, node_color="lightblue", edgecolors="red", node_size=700, linewidths=2, ax=ax)

plt.show()
