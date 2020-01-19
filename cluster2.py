import markov_clustering as mc
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

network = nx.Graph()

f1 = open('data.csv','r')
Peopledata = f1.read().split('\n')
PeopleNodes = []
k = 1
peopleEdge = []

iis = []
for person in Peopledata:
    if len(person)>0:
        person = person.split(',')
        if person[2] !='':
            u = person[2].split(' ')
            m = 0
            for i in u:
                m += 1
                peopleEdge.append([int(person[0]),int(i), 1/(m**k)])
            iis.append((person[0],person[1]))


delist = []

for j in range(len(peopleEdge)):
    for i in range(len(peopleEdge)-j):
        if(peopleEdge[j][0] == peopleEdge[j+i][1] and peopleEdge[j][1] == peopleEdge[j+i][0]):
            peopleEdge[j][2] += peopleEdge[j+i][2]
            delist.append(j+i)

delist.sort(reverse = True)

j = 0
for i in delist:
    if i != j:
        try:
            del peopleEdge[i]
        except:
            continue
    j = i


matinput = []

for edge in peopleEdge:

    if edge[0] not in matinput:
        matinput.append(edge[0])
    if edge[1] not in matinput and edge[1] !=0 :
        matinput.append(edge[1])

    network.add_edge(edge[0],edge[1], weight = edge[2])

network.remove_node(0)

print(peopleEdge)
print(matinput)
matrix = nx.to_scipy_sparse_matrix(network)

print(matrix)

result = mc.run_mcl(matrix)           # run MCL with default parameters
print(result)
clusters = mc.get_clusters(result)
print(clusters)
pos = {}
spos = [(0,0),(0,1),(1,0),(1,1),(2,1),(2,2),(1,2),(0,2)] #spatial positioning for clustered set. If needed, add more!!

ctr = 0

network = nx.from_scipy_sparse_matrix(result)
network2 = nx.from_scipy_sparse_matrix(matrix)

for g in clusters:
    r = {i:(random.random()+spos[ctr][0],random.random()+spos[ctr][1]) for i in g}
    pos.update(r)
    ctr += 1

colors = []

cluster_map = {node: i for i, cluster in enumerate(clusters) for node in cluster}
colors = [cluster_map[i] for i in range(len(network.nodes()))]

nx.draw_networkx_nodes(network, pos, node_color = colors,node_size = 200)
nx.draw_networkx_edges(network2, pos, width = 0.5, style = 'dashed')
nx.draw_networkx_edges(network, pos, width=2.0, alpha=0.5)

print(network2.get_edge_data(0,1)['weight'])

plt.axis('off')
plt.show()

wwclist = []
for g in clusters:
    for v in g:
        sum1 = 0
        sum2 = 0
        N = list(network2.neighbors(v))
        N1 = N.copy()
        N1.extend(g)
        N1 = list(set(list(N1)))
        N2 = [v for v in g if v in N]
        for v2 in N1:
            try:
                sum1 += network2.get_edge_data(v,v2)['weight']
            except:
                continue
        for v2 in N2:
            try:
                sum2 += network2.get_edge_data(v,v2)['weight']
            except:
                continue
        wwclist.append((matinput[v], sum2/sum1))


wwclist.sort()
print(wwclist)
iis.sort()
data = []
print(iis)
for i in range(len(iis)):
    data.append([int(iis[i][1])/7,wwclist[i][1]])

data2d = np.array(list(data)).T

y = data2d[0]
x = data2d[1]
z = np.sqrt(x**2+y**2)

plt.subplot(321)
plt.scatter(x, y, s=36, c=(0.2, 0.2, 0.2, 0.3), marker="+")

plt.show()

print(np.corrcoef(y,x))
