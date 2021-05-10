import math

def thefunction(x,sortedlistofweights,fuzzyvalues): # the list of weights have to be sorted
    if x > len(sortedlistofweights):
        return "Input not in range!"
    if x == 0:
        return 0
    sum = 0
    for item in fuzzyvalues:
        j = 0
        while j < x:
            if item[sortedlistofweights[j][1]] == 1 and sortedlistofweights[j][0] > 0.98:
                A = 1
            else:
                A = 0
            j += 1
        sum = sum + A
    return sum




gr = (math.sqrt(5) - 1) / 2

def gss(f, a, b, tol = .01):
    c = b - (b - a) * gr
    d = a + (b - a) * gr
    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) * gr
        d = a + (b - a) * gr

    return (b + a) / 2



