import csv, os
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3, 4, 5, 6])

with open(os.path.join('results', 'FGSM.csv')) as file:
    csv_reader = csv.reader(file, delimiter=',')
    line_count = 0
    row_count = 0
    for row in csv_reader:
        for idx in range(len(row)):
            if idx < row_count:
                continue

            if line_count == row_count:
                row_count += 1
                continue

            try:
                column = float(row[idx])
            except:
                print(
                    'column [' + row[idx] + '] in ' + str(line_count) + ', ' + str(row_count) + ' is not a number')

            weight = 1 / float(column)
            G.add_edge(line_count, row_count, weight=weight)
            row_count += 1

        line_count += 1
        row_count = line_count

pos = nx.fruchterman_reingold_layout(G, iterations=500)
nx.draw(G, pos, with_labels=True, font_weight='bold')
#nx.draw_networkx_edge_labels(G, pos)
plt.show()

# # color by path length from node near center
# p = dict(nx.single_source_shortest_path_length(G, ncenter))
#
# plt.figure(figsize=(8, 8))
# nx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4)
# nx.draw_networkx_nodes(G, pos, nodelist=list(p.keys()),
#                        node_size=80,
#                        node_color=list(p.values()),
#                        cmap=plt.cm.Reds_r)
#
# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
# plt.axis('off')
# plt.show()