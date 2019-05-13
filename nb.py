# Naive Bayes' Classifier

import sys
training_set = [
[1,1,2,'a'],
[2,1,1,'a'],
[2,0,1,'a'],
# [2,2,2,'b'], # New
[0,2,1,'b'],
[3,2,0,'b'],
[3,3,0,'c'],
[0,3,0,'c'],
[3,2,1,'c'],
[0,3,3,'c']
]

def getCatCount(cat):
    count = 0
    for d in training_set:
        if cat == d[3]: count += 1
    return count

def notCatCount(cat):
    count = 0
    for d in training_set:
        if cat != d[3]: count += 1
    return count

def jointCatAttr(cat, attr, val):
    count = 0
    for d in training_set:
        if d[attr-1] == val and d[3] == cat: count += 1 
    return count

def jointNotCat(cat, attr, val):
    count = 0
    for d in training_set:
        if cat != d[3] and d[attr-1] == val: count += 1
    return count

def jointAttr(testA1, testA2, testA3):
    total = 0
    testA1count = 0
    testA2count = 0
    testA3count = 0
    for d in training_set:
        if d[0]==testA1: testA1count += 1
        if d[1]==testA2: testA2count += 1
        if d[2]==testA3: testA3count += 1
        total += 1
    pta1 = testA1count / total
    pta2 = testA2count / total
    pta3 = testA3count / total
    return pta1 * pta2 * pta3

def pAttrICat(countJoint, countCat, laplace, k):
    numer = (countJoint + laplace)
    denom = (countCat + (k*laplace))
    return numer/denom

#   P(A1|C)P(A2|C)P(A3|C)P(C)
# P(A1|C)P(A2|C)P(A3|C)P(C) + P(A1|~C)P(A2|~C)P(A3|~C)P(~C)

def probCatIAttr(cat,testA1,testA2,testA3,laplace,k):
    catCount = getCatCount(cat)
    nCatCount = notCatCount(cat)
    total = catCount + nCatCount
    # Conditional probabilities
    a1ICat = pAttrICat( jointCatAttr(cat,1,testA1), catCount, laplace, k)
    a2ICat = pAttrICat( jointCatAttr(cat,2,testA2), catCount, laplace, k)
    a3ICat = pAttrICat( jointCatAttr(cat,3,testA3), catCount, laplace, k)
    # Complementary probabilities
    a1InotCat = pAttrICat( jointNotCat(cat, 1, testA1), nCatCount, laplace, k)
    a2InotCat = pAttrICat( jointNotCat(cat, 2, testA1), nCatCount, laplace, k)
    a3InotCat = pAttrICat( jointNotCat(cat, 3, testA1), nCatCount, laplace, k)
    # Result
    numer = a1ICat*a2ICat*a3ICat*(catCount/total)
    denom = numer + (a1InotCat*a2InotCat*a3InotCat*(nCatCount/total))
    # denom = jointAttr(testA1,testA2,testA3)
    return numer/denom

inp = sys.argv[1:4]
inp = [int(x) for x in inp]
laplace=0.1
k=4
paIu = probCatIAttr("a",*inp,laplace,k)
pbIu = probCatIAttr("b",*inp,laplace,k)
pcIu = probCatIAttr("c",*inp,laplace,k)
print(paIu)
print(pbIu)
print(pcIu)

# def derive_conditional_data(_tset, _cat):
#     a1,a2,a3 = {},{},{}
#     cat_count = 0
#     for datum in _tset:
#         cat = datum[3]
#         if cat == _cat:
#             cat_count += 1
#             a1col = datum[0]
#             a2col = datum[1]
#             a3col = datum[2]
#             a1[a1col] = a1.get(a1col, 0) + 1
#             a2[a2col] = a2.get(a2col, 0) + 1
#             a3[a3col] = a3.get(a3col, 0) + 1
#     return _cat, (a1,a2,a3,cat_count)

# def derive_conditionals(_conditional_data):
#     delt = 0.1 # Laplacian Corrector
#     k = 4
#     count = _conditional_data[3]
#     cd = _conditional_data[0:3]
#     A1givenC, A2givenC, A3givenC = {}, {}, {}
#     a1_data = cd[0]
#     a2_data = cd[1]
#     a3_data = cd[2]
#     for occurences in a1_data:
#         A1givenC[occurences] = float( (a1_data[occurences]+delt) / (count + k*delt) )
#     for occurences in a2_data:
#         A2givenC[occurences] = float( (a2_data[occurences]+delt) / (count + k*delt) )
#     for occurences in a3_data:
#         A3givenC[occurences] = float( (a3_data[occurences]+delt) / (count + k*delt) )
#     # print('A1',A1givenC,'\nA2',A2givenC,'\nA3',A3givenC)
#     default = float( (delt) / (count + k*delt) )
#     return (A1givenC, A2givenC, A3givenC), default

# def predict(A1,A2,A3,conds,pClass,defaults):
#     for key in conds:
#         pA1gKey = conds[key][0].get(A1, defaults[key])
#         pA2gKey = conds[key][1].get(A2, defaults[key])
#         pA3gKey = conds[key][2].get(A3, defaults[key])
#         numer = pA1gKey*pA2gKey*pA3gKey*pClass[key]
#         pA1gNotKey = 0
#         pA2gNotKey = 0
#         pA3gNotKey = 0
#         for notkey in conds:
#             if notkey != key:
#                 pA1gNotKey += conds[notkey][0].get(A1, defaults[notkey])
#                 pA2gNotKey += conds[notkey][1].get(A2, defaults[notkey])
#                 pA3gNotKey += conds[notkey][2].get(A3, defaults[notkey])
#         # print(pA1gKey, pA1gNotKey)
#         denom = numer + pA1gNotKey*pA2gNotKey*pA3gNotKey*(1-pClass[key])
#         print("Probability of {key} is {prob}".format(key=key, prob=numer/denom))

# conds = {}
# defaults = {}
# for _cat in ['a','b','c']:
#     cat, ca = derive_conditional_data(training_set, _cat)
#     # print(cat)
#     conds[_cat], defaults[_cat] = derive_conditionals(ca)
# pClass = {'a': 3/9, 'b': 2/9, 'c': 4/9}
# print(conds)
# inp = sys.argv[1:4]
# inp = [int(x) for x in inp]
# predict(*inp,conds,pClass,defaults)
