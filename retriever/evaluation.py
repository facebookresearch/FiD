import numpy as np

def eval_batch(scores, inversions, avg_topk, idx_topk):
    for k, s in enumerate(scores):
        s = s.cpu().numpy()
        sorted_idx = np.argsort(-s)
        score(sorted_idx, inversions, avg_topk, idx_topk)

def count_inversions(arr): 
    inv_count = 0
    lenarr = len(arr)
    for i in range(lenarr): 
        for j in range(i + 1, lenarr): 
            if (arr[i] > arr[j]): 
                inv_count += 1
    return inv_count

def score(x, inversions, avg_topk, idx_topk):
    x = np.array(x)
    inversions.append(count_inversions(x))
    for k in avg_topk:
        avg_pred_topk = (x[:k]<k).mean() #ratio of passages in the predicted top-k that are also in the topk given by gold score
        avg_topk[k].append(avg_pred_topk)
    for k in idx_topk:
        below_k = (x<k)
        idx_gold_topk = len(x) - np.argmax(below_k[::-1]) #number of passages required to obtain all passages from gold top-k
        idx_topk[k].append(idx_gold_topk)
    
    