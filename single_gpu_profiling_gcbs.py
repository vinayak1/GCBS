#Env setup                                                                                                                                                                                                         
# docker run --gpus device=all -it --rm --shm-size 16gb -v /home/cust_admin/:/repo/ -it nvcr.io/nvidia/pytorch:22.05-py3 bash

import torch
import timeit

def compute_gce(z1_outs, z2_outs, quantile):                                                                                                                                                                                                                                                                                                                           
      import math                                                                                                                                                                                                  
      from scipy.sparse.csgraph import reverse_cuthill_mckee                                                                                                                                                       
      from scipy.sparse import csr_matrix                                                                                                                                                                          
      import statistics
      start_time = timeit.default_timer()
      # (1) Stack and normalize outputs                                                                                                                                                                            
      src_train_full = torch.nn.functional.normalize(z1_outs).cuda()                                                                                                                                 
      tgt_train_full = torch.nn.functional.normalize(z2_outs).cuda()                                                                                                                                 
      z1_outs, z2_outs = [], []                                                                                                                                                                                    
                                                                                                                                                                  
      # (2) Estimate quantile                                                                                                                                                                                      
      chunk_size, num_samples, quantiles = 3, len(tgt_train_full), []                                                                                                                                             
      for chunk_idx in range(math.ceil(len(tgt_train_full)/chunk_size)):                                                                                                                                           
        mat_val = src_train_full[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size] @ tgt_train_full.T                                                                                                                 
        quantiles.append(float(torch.quantile(mat_val, quantile)))                                                                                                                                                 
                                                                                                                       
      # (3) Get similarity graph thresholded on quantile                                                                                                                                                           
      row, col, data, quantile = [], [], [], statistics.median(quantiles)                                                                                                                                          
      for chunk_idx in range(math.ceil(len(src_train_full)/chunk_size)):                                                                                                                                           
        mat_val = src_train_full[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size] @ tgt_train_full.T                                                                                                                 
        ret = (mat_val.flatten() > quantile).nonzero(as_tuple=True)[0].cpu()                                                                                                                                       
        row += ((ret - (ret % num_samples))/num_samples + chunk_idx*chunk_size).int().tolist()                                                                                                                     
        col += (ret % num_samples).tolist()                                                                                                                                                                        
        data += [1.0 for _ in range(len(ret))]                                                                                                                                                                     
                                                                                                                                             
      # (4) Get permutation using graph bandwidth minimization on sparsified graph (cuthill-mckee)                                                                                                                 
      permutation = list(reverse_cuthill_mckee(csr_matrix((data, (row, col)),                                                                                                                                      
                                                    shape=(num_samples, num_samples)))) 
      print(timeit.default_timer() - start_time)
      return permutation

for size in [10000, 50000, 100000, 500000, 1000000]:
    keep_per_sample = 512
    z1, z2 = torch.rand(size, 768), torch.rand(size, 768)
    quantile = 1 - float(keep_per_sample/size)
    perm = compute_gce(z1, z2, quantile)