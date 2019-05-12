# Naive Bayes' Classifier

training_set = [
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

def derive_conditional_data(_tset, _cat):
    a1,a2,a3 = {},{},{}
    cat_count = 0
    for datum in _tset:
        cat = datum[3]
        if cat == _cat:
            cat_count += 1
            a1col = datum[0]
            a2col = datum[1]
            a3col = datum[2]
            a1[a1col] = a1.get(a1col, 0) + 1
            a2[a2col] = a2.get(a2col, 0) + 1
            a3[a3col] = a3.get(a3col, 0) + 1
    return _cat, (a1,a2,a3,cat_count)

def derive_conditionals(_conditional_data):
    delt = 0.1 # Laplacian Corrector
    k = 4
    count = _conditional_data[3]
    cd = _conditional_data[0:3]
    A1givenC, A2givenC, A3givenC = {}, {}, {}
    a1_data = cd[0]
    a2_data = cd[1]
    a3_data = cd[2]
    for occurences in a1_data:
        A1givenC[occurences] = float( (a1_data[occurences]+delt) / (count + k*delt) )
    for occurences in a2_data:
        A2givenC[occurences] = float( (a2_data[occurences]+delt) / (count + k*delt) )
    for occurences in a3_data:
        A3givenC[occurences] = float( (a3_data[occurences]+delt) / (count + k*delt) )
    # print('A1',A1givenC,'\nA2',A2givenC,'\nA3',A3givenC)
    default = float( (delt) / (count + k*delt) )
    return (A1givenC, A2givenC, A3givenC), default

def predict(A1,A2,A3,conds,pClass,defaults):
    for key in conds:
        pA1gKey = conds[key][0].get(A1, defaults[key])
        pA2gKey = conds[key][1].get(A2, defaults[key])
        pA3gKey = conds[key][2].get(A3, defaults[key])
        numer = pA1gKey*pA2gKey*pA3gKey*pClass[key]
        pA1gNotKey = 0
        pA2gNotKey = 0
        pA3gNotKey = 0
        for notkey in conds:
            if notkey != key:
                pA1gNotKey += conds[notkey][0].get(A1, defaults[notkey])
                pA2gNotKey += conds[notkey][1].get(A2, defaults[notkey])
                pA3gNotKey += conds[notkey][2].get(A3, defaults[notkey])
        # print(pA1gKey, pA1gNotKey)
        denom = numer + pA1gNotKey*pA2gNotKey*pA3gNotKey*(1-pClass[key])
        print("Probability of {key} is {prob}".format(key=key, prob=numer/denom))

conds = {}
defaults = {}
for _cat in ['a','b','c']:
    cat, ca = derive_conditional_data(training_set, _cat)
    # print(cat)
    conds[_cat], defaults[_cat] = derive_conditionals(ca)
pClass = {'a': 3/9, 'b': 2/9, 'c': 4/9}
print(conds)
predict(2,2,2,conds,pClass,defaults)
