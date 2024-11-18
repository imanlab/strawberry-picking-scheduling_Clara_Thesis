from scipy import stats

def find_lowest_kendall_tau(arrays):
   min_tau = float('inf')
   best_pair = None
   for i in range(len(arrays)):
       for j in range(i + 1, len(arrays)):
           tau, _ = stats.kendalltau(arrays[i], arrays[j])
           if abs(tau) < min_tau:
               min_tau = abs(tau)
               best_pair = (arrays[i], arrays[j])
   return best_pair

