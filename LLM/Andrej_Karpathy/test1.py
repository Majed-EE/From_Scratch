sides=[4,6,8,12,20]
no_of_h=len(sides)
D=[4,6,8,12,20]
H = list([1/no_of_h for x in range(no_of_h) ]) # prior probability assuming flat prior
# D=[4,6,8,12,20]
data=D[3]
# likelihoods = list(H[x]*1/sides[x] if sides[x] >= data else 0 for x in range(len(H)))
likelihoods = []
for i in range(len(H)):
    side=1/sides[i]
    if sides[i] >= data:
        prior=H[i]
        likelihood=prior*side
        likelihoods.append(H[i]*1/sides[i])
    else:
        prior=0
        likelihood=0
        likelihoods.append(0)
print(likelihoods)