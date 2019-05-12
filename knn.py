# K Nearest Neighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
K = 3

training = [
[1,1,2,'a'],
[2,1,1,'a'],
[2,0,1,'a'],
[0,2,1,'b'],
[3,2,0,'b'],
[3,3,0,'c'],
[0,3,0,'c'],
[3,2,1,'c'],
[0,3,3,'c']
]

def distance(a0,b0,c0,a,b,c):
    sub = (a-a0)**2 + (b-b0)**2 + (c-c0)**2
    res = sub**0.5
    return res

def dists(a,b,c):
    distances = {}
    for i, datum in enumerate(training): 
        distances[i] = distance(*datum[0:3],a,b,c)
    return distances

def nearest(distances, set):
    sorted_dists = sorted(distances.items(), key=lambda x: x[1])
    return sorted_dists
         
test = (2,2,2)
distances = dists(*test)
res = nearest(distances, training)
knearest = {}
for i, tup in enumerate(res):
    ind = tup[0]
    cl = training[ind][3]
    knearest[cl] = knearest.get(cl, 0) + 1 
    if i+1 == K: break
print(knearest)
axs = array([datum[0] for datum in training if datum[3]=="a"])
ays = array([datum[1] for datum in training if datum[3]=="a"]) 
azs = array([datum[2] for datum in training if datum[3]=="a"]) 
bxs = array([datum[0] for datum in training if datum[3]=="b"])
bys = array([datum[1] for datum in training if datum[3]=="b"]) 
bzs = array([datum[2] for datum in training if datum[3]=="b"]) 
cxs = array([datum[0] for datum in training if datum[3]=="c"])
cys = array([datum[1] for datum in training if datum[3]=="c"]) 
czs = array([datum[2] for datum in training if datum[3]=="c"]) 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(axs, ays, azs,c="red",s=50)
ax.scatter(bxs, bys, bzs,c="blue",s=50)
ax.scatter(cxs, cys, czs,c="orange",s=50)
ax.scatter(array([2]), array([2]), array([2]),c="green",s=65)
plt.show()
