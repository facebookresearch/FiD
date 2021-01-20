import numpy as np

def count_inversions(arr): 
    inv_count = 0
    lenarr = len(arr)
    for i in range(lenarr): 
        for j in range(i + 1, lenarr): 
            if (arr[i] > arr[j]): 
                inv_count += 1
    return inv_count

def score(x, metric_at):
    el_topm, el_avgm = {}, {}
    inversions = count_inversions(x)
    x = np.array(x)
    for k in metric_at:
        t = (x[:k]<k).mean()
        a = (x<k)
        a = len(x) - np.argmax(a[::-1])
        el_topm[k] = t
        el_avgm[k] = a
    return inversions, el_topm, el_avgm
    
    