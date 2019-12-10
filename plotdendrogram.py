import csv, os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd

ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
                    400., 754., 564., 138., 219., 869., 669.])


with open(os.path.join('results', '2-CW_left.csv')) as file:
    csv_reader = csv.reader(file, delimiter=',')
    res = []

    line_count = 0
    row_count = 0
    for row in csv_reader:
        line = []
        for column in row:

            try:
                column = float(column)
            except:
                print(
                    'column [' + column + '] in ' + str(line_count) + ', ' + str(row_count) + ' is not a number')
            line.append(column)
            row_count += 1
        res.append(line)
        row_count = 0
        line_count += 1

res = np.asarray(res)
distArray = ssd.squareform(res)

hierarchy.set_link_color_palette(['#003361'])

Z = hierarchy.linkage(distArray, 'single')
plt.figure()
dn = hierarchy.dendrogram(Z)

dn1 = hierarchy.dendrogram(Z, above_threshold_color='#f39200', orientation='top')
plt.show()