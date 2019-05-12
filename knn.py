# K Nearest Neighbors

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