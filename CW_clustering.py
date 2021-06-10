import time
import scipy
import numpy as np
import pandas as pd
from pdb import set_trace
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn import preprocessing
from matplotlib.ticker import MaxNLocator

fig1, ax1 = plt.subplots()

ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

factor = 10.0

U = np.load('/Users/soumianarayanan.vija/Programs/ML/Final_hyper_elastoplasticity/Grid_approach/Simulations/Monotonic_loading_simulations/Training_results/Disp_vals/all_disp_vals.npy')

#to_cluster = np.concatenate((A, B), axis=0)

U_remaining = np.copy(U.transpose())

'##Remove variables with zeros'
nr_var = len(U_remaining[0, :])
nr_points = len(U_remaining[:, 0])
rem = []
for i in range(nr_var):
    if min(U_remaining[:, i]) > max(U_remaining[:, i])-1.0e-10:
        rem.append(i)

U_remaining = np.delete(U_remaining, rem, 1)
nr_var = U_remaining.shape[1]

'#--------------------------------------------------------------------#'
'#normalise snapshots#'
#for i in range(nr_points):
#    u = U_remaining[i, :]
#    U_remaining[i, :] = u/np.linalg.norm(u)
#    print(i,np.linalg.norm(u))

U_remaining = preprocessing.normalize(U_remaining,norm='l2',axis=1)


'#measure mutual distances#'
"""
DISTANCES = np.zeros((nr_points, nr_points))
for i in range(nr_points-1):
    ui = U_remaining[i, :]
    for j in range(i+1, nr_points):
        uj = U_remaining[j, :]
        distance = np.linalg.norm(ui-uj)
        DISTANCES[i, j] = distance
        DISTANCES[j, i] = distance
"""

DISTANCES = cdist(U_remaining, U_remaining)


numbering = np.zeros((nr_points, 1), dtype='int32')
nrs_per_cluster = np.zeros((len(U_remaining[:, 0]), 1), dtype='int32')

remaining_numbers = np.linspace(0, nr_points-1, nr_points, dtype='int32')

cluster_counter = 0
while len(U_remaining[:, 0]) > 1:

    cluster_counter += 1

    nr_points = len(U_remaining[:, 0])

    DISTANCES_ranked = np.sort(DISTANCES, axis=1)
    DISTANCES_ranked_nrs = np.argsort(DISTANCES, axis=1)
    radius = np.mean(DISTANCES_ranked[:, 1])*factor

    densities = np.zeros((nr_points, 1))

    for i in range(nr_points):
        a, b = np.where(DISTANCES_ranked[i, :].reshape(1, nr_points) < radius)
        densities[i] = sum(np.exp(-5*DISTANCES_ranked[i, b[1:len(b)]]))

    cluster_numbering = np.zeros((nr_points, 1), dtype='int32')
    b = np.argmax(densities)
    cluster_numbering[b] = 1
    YN = 0
    while YN == 0:
        cluster_numbering_previous = np.copy(cluster_numbering)
        a = np.where(cluster_numbering == 1)[0]

        for i in range(len(a)):
            i = a[i]
            c, d = np.where(DISTANCES_ranked[i, :].reshape(1, nr_points) < radius)
            cluster_numbering[DISTANCES_ranked_nrs[i, d[1:len(d)]]] = 1

        if sum(cluster_numbering) == sum(cluster_numbering_previous):
            YN = 1
    U_cluster = np.copy(U_remaining[np.where(cluster_numbering == 1)[0], :])
    U_remaining = np.copy(U_remaining[np.where(cluster_numbering == 0)[0], :])

    rem = np.linspace(0, nr_points-1, nr_points, dtype='int32')
    rem = rem[np.where(cluster_numbering == 1)[0]]

    DISTANCES = np.delete(DISTANCES, rem, 0)
    DISTANCES = np.delete(DISTANCES, rem, 1)

    removed_numbers = remaining_numbers[np.where(cluster_numbering == 1)[0]]
    remaining_numbers = remaining_numbers[np.where(cluster_numbering == 0)[0]]

    numbering[removed_numbers] = cluster_counter
    nrs_per_cluster[cluster_counter-1] = sum(cluster_numbering)

set_trace()
if len(U_remaining) != 0:
    cluster_counter = cluster_counter+1
    nrs_per_cluster[cluster_counter-1] = 1
    numbering[remaining_numbers] = cluster_counter
    U_remaining = []

nrs_per_cluster = np.delete(nrs_per_cluster, range(cluster_counter, len(nrs_per_cluster)))

print(nrs_per_cluster)
print(nrs_per_cluster.shape)

set_trace()

nr_clus = nrs_per_cluster.shape[0]
XX = pd.DataFrame(U.transpose())

cluster_map = pd.DataFrame()
cluster_map['data_index'] = XX.index.values
cluster_map['cluster'] = numbering

cl1 = cluster_map[cluster_map.cluster == 1]
cl2 = cluster_map[cluster_map.cluster == 2]
cl3 = cluster_map[cluster_map.cluster == 3]
cl4 = cluster_map[cluster_map.cluster == 4]
cl5 = cluster_map[cluster_map.cluster == 5]
cl6 = cluster_map[cluster_map.cluster == 6]
cl7 = cluster_map[cluster_map.cluster == 7]
cl8 = cluster_map[cluster_map.cluster == 8]
cl9 = cluster_map[cluster_map.cluster == 9]
cl10 = cluster_map[cluster_map.cluster == 10]
cl11 = cluster_map[cluster_map.cluster == 11]

cl1_data = XX.iloc[cl1.data_index, :]
cl2_data = XX.iloc[cl2.data_index, :]
cl3_data = XX.iloc[cl3.data_index, :]
cl4_data = XX.iloc[cl4.data_index, :]
cl5_data = XX.iloc[cl5.data_index, :]
cl6_data = XX.iloc[cl6.data_index, :]
cl7_data = XX.iloc[cl7.data_index, :]
cl8_data = XX.iloc[cl8.data_index, :]
cl9_data = XX.iloc[cl9.data_index, :]
cl10_data = XX.iloc[cl10.data_index, :]
cl11_data = XX.iloc[cl11.data_index, :]


print(cl1_data.shape)
print(cl2_data.shape)
print(cl3_data.shape)
print(cl4_data.shape)
print(cl5_data.shape)

print(cl6_data.shape)
print(cl7_data.shape)
print(cl8_data.shape)
print(cl9_data.shape)
print(cl10_data.shape)

u_fluc_cl1 = cl1_data.transpose()
u1, s1, vt1 = svds(u_fluc_cl1,k=45)

uu1 = np.flip(u1,axis=1)

xvalues1 = np.linspace(0,len(s1),len(s1))
yvalues1 = np.flip(s1,axis=0)
yyvalues1 = yvalues1/yvalues1[0]

#
ax1.plot(xvalues1, yyvalues1,color='k',label = "Cluster 1", linewidth=2)
#
ax1.legend(loc="upper right")
ax1.set_yscale('log')
plt.yscale('log')


u_fluc_cl2 = cl2_data.transpose()
u2, s2, vt2 = svds(u_fluc_cl2,k=18)

uu2 = np.flip(u2,axis=1)


xvalues2 = np.linspace(0,len(s2),len(s2))
yvalues2 = np.flip(s2,axis=0)
yyvalues2 = yvalues2/yvalues2[0]

ax1.plot(xvalues2, yyvalues2,color='r',label = "Cluster 2", linewidth=2)
ax1.legend(loc="upper right")
ax1.set_yscale('log')

plt.show()
