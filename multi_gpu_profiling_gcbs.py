# From Jingyun Liang's script                       
# #Env setup                                                                                                                                                                                                         
# docker run --gpus device=all -it --rm --shm-size 16gb -v /home/cust_admin/:/repo/ -it nvcr.io/nvidia/pytorch:22.05-py3 bash                                                                                                                                                               
# --------------------------------------------------------                                                                                                                                                         
import timeit                                                                                                                                                                                                      
import math                                                                                                                                                                                                        
import os                                                                                                                                                                                                          
import random                                                                                                                                                                                                      
import time                                                                                                                                                                                                        
import warnings                                                                                                                                                                                                    
import torch                                                                                                                                                                                                       
import multiprocessing                                                                                                                                                                                             
import itertools                                                                                                                                                                                                   
from scipy.sparse.csgraph import reverse_cuthill_mckee                                                                                                                                                             
from scipy.sparse import csr_matrix                                                                                                                                                                                
import statistics                                                                                                                                                                                                  
                                                                                                                                                                                                                   
# global variables                                                                                                                                                                                                 
ngpus = 7                                                                                                                                                                                                          
total_gpu_num = 7                                                                                                                                                                                                  
# max_process_per_gpu=1 always works fine; max_process_per_gpu>=2 may get stuck on some computers or clusters.                                                                                                     
max_process_per_gpu = 1                                                                                                                                                                                            
used_gpu_list = multiprocessing.Manager().list([0] * total_gpu_num)                                                                                                                                                
lock = multiprocessing.Lock()                                                                                                                                                                                      
                                                                                                                                                                                                                   
def quantile_estimation(src_mat, tgt_mat, quantile):                                                                                                                                                                                                                                                                                                                                                                     
    chunk_size = 3                                                                                                                                                                                                                                                                                                                                                                                                       
    slice_quantiles = []                                                                                                                                                                                           
    for chunk_idx in range(math.ceil(len(src_mat)/chunk_size)):                                                                                                                                                    
      mat_val = src_mat[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size] @ tgt_mat.T                                                                                                                                 
      slice_quantiles.append(float(torch.quantile(mat_val, quantile)))                                                                                                                                             
                                                                                                                                                                                                                   
    return slice_quantiles                                                                                                                                                                                         
                                                                                                                                                                                                                   
def get_thresh_sim_graph(src_mat, tgt_mat, quantile):                                                                                                                                                                        
    row, col, data = [], [], []                                                                                                                                                                                    
    chunk_size = 3            
    num_samples = len(tgt_mat)                                                                                                                                                                                    
    for chunk_idx in range(math.ceil(len(src_mat)/chunk_size)):                                                                                                                                                    
        mat_val = src_mat[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size] @ tgt_mat.T                                                                                                                               
        ret = (mat_val.flatten() > quantile).nonzero(as_tuple=True)[0].cpu()                                                                                                                                       
        row.extend(((ret - (ret % num_samples))/num_samples + chunk_idx*chunk_size).int().tolist())                                                                                                                
                                                                                                                                                                                                                   
        col.extend((ret % num_samples).tolist())                                                                                                                                                                   
                                                                                                                                                                                                                   
        data.extend([1.0 for _ in range(len(ret))])                                                                                                                                                                
    return (row, col, data)     

def multi_gpu_testing_wrapper(input, input2, quantile, index, gpu_id=None, available_gpu_num=1):                                                                                                                                                                                                                                                                                                                               
    '''                                                                                                                                                                                                            
    Multi-GPU testing wrapper.                                                                                                                                                                                     
    :param model: The PyTorch model (on cpu, NOT on cuda).                                                                                                                                                         
    :param input: The model input, e.g. an image (on cpu, NOT on cuda).                                                                                                                                            
    :param index: Sample index (int).                                                                                                                                                                              
    :param gpu_id: Given gpu_id. Only used in debugging.                                                                                                                                                           
    :param available_gpu_num: Available GPU number. Default: 1.                                                                                                                                                    
    :return: Model output (on cpu, NOT on cuda), used GPU id and process id.                                                                                                                                       
    '''                                                                                                                                                                                                            
    # GPU assignment                                                                                                                                                                                               
    lock.acquire()                                                                                                                                                                                                 
    if gpu_id is None:                                                                                                                                                                                             
        for i in range(available_gpu_num):                                                                                                                                                                         
            if used_gpu_list[i] < max_process_per_gpu:                                                                                                                                                             
                gpu_id = i                                                                                                                                                                                         
                break                                                                                                                                                                                              
                                                                                                                                                                                                                   
    used_gpu_list[gpu_id] += 1                                                                                                                                                                                     
    lock.release()                                                                                                                                                                                                 
    torch.cuda.set_device(gpu_id)                                                                                                                                                                                  
    device = torch.device('cuda')                                                                                                                                                                                  
    print(f'testing   input {index} on GPU {gpu_id}. Overall GPU usages: ', list(used_gpu_list))                                                                                                                   
                                                                                                                                                                                                                   
    # model testing                                                                                                                                                                                                
    input = input.to(device)                                                                                                                                                                                       
    input2 = input2.to(device)                                                                                                                                                                                     
    time.sleep(random.randrange(0, 10)) # used in this toy example to avoid deadlock                                                                                                                               
                                                                                                                                                                                                                   
    output = quantile_estimation(input,input2,quantile)                                                                                                                                                            
                                                                                                                                                                                                                   
                                                                                                                                                                                                                   
    # release GPU memory manually (multiprocessing.Pool may not release GPU memory of a process in time)                                                                                                           
    del input, input2                                                                                                                                                                                              
    torch.cuda.empty_cache()                                                                                                                                                                                       
                                                                                                                                                                                                                   
    # release GPU                                                                                                                                                                                                  
    lock.acquire()                                                                                                                                                                                                 
    used_gpu_list[gpu_id] -= 1                                                                                                                                                                                     
    lock.release()                                                                                                                                                                                                 
    print(f'releasing input {index} on GPU {gpu_id}. Overall GPU usages: ', list(used_gpu_list))                                                                                                                   
    # return output                                                                                                                                                                                                
    return output, gpu_id, os.getpid()                                                                                                                                                                             
                                                                                                                                                                                                                   
                                                                                                                                                                                                                   
def multi_gpu_testing_wrapper2(input, input2, quantile, index, gpu_id=None, available_gpu_num=1):                                                                                                                            
                                                                                                                                                                                                                   
    '''                                                                                                                                                                                                            
    Multi-GPU testing wrapper.                                                                                                                                                                                     
    :param model: The PyTorch model (on cpu, NOT on cuda).                                                                                                                                                         
    :param input: The model input, e.g. an image (on cpu, NOT on cuda).                                                                                                                                            
    :param index: Sample index (int).                                                                                                                                                                              
    :param gpu_id: Given gpu_id. Only used in debugging.                                                                                                                                                           
    :param available_gpu_num: Available GPU number. Default: 1.                                                                                                                                                    
    :return: Model output (on cpu, NOT on cuda), used GPU id and process id.                                                                                                                                       
    ''' 


    # GPU assignment                                                                                                                                                                                      [57/1996]
    lock.acquire()                                                                                                                                                                                                 
    if gpu_id is None:                                                                                                                                                                                             
        for i in range(available_gpu_num):                                                                                                                                                                         
            if used_gpu_list[i] < max_process_per_gpu:                                                                                                                                                             
                gpu_id = i                                                                                                                                                                                         
                break                                                                                                                                                                                              
                                                                                                                                                                                                                   
    used_gpu_list[gpu_id] += 1                                                                                                                                                                                     
    lock.release()                                                                                                                                                                                                 
    torch.cuda.set_device(gpu_id)                                                                                                                                                                                  
    device = torch.device('cuda')                                                                                                                                                                                  
    print(f'testing   input {index} on GPU {gpu_id}. Overall GPU usages: ', list(used_gpu_list))                                                                                                                   
                                                                                                                                                                                                                   
    # model testing                                                                                                                                                                                                
    input = input.to(device)                                                                                                                                                                                       
    input2 = input2.to(device)                                                                                                                                                                                     
    time.sleep(random.randrange(0, 10)) # used in this toy example to avoid deadlock                                                                                                                               
                                                                                                                                                                                                                   
    output = get_thresh_sim_graph(input,input2, quantile)                                                                                                                                                                    
                                                                                                                                                                                                                   
                                                                                                                                                                                                                   
    # release GPU memory manually (multiprocessing.Pool may not release GPU memory of a process in time)                                                                                                           
    del input, input2                                                                                                                                                                                              
    torch.cuda.empty_cache()                                                                                                                                                                                       
                                                                                                                                                                                                                   
    # release GPU                                                                                                                                                                                                  
    lock.acquire()                                                                                                                                                                                                 
    used_gpu_list[gpu_id] -= 1                                                                                                                                                                                     
    lock.release()                                                                                                                                                                                                 
    print(f'releasing input {index} on GPU {gpu_id}. Overall GPU usages: ', list(used_gpu_list))                                                                                                                   
    # return output                                                                                                                                                                                                
    return output, gpu_id, os.getpid()                                                                                                                                                                             
                                                                                                                                                                                                                   
                                                                                                                                                                                                                   
def main():                                                                                                                                                                                                        
    # setup GPU                                                                                                                                                                                                    
    available_gpu_num = 7                                                                                                                                                                                          
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"                                                                                                                                                           
    assert available_gpu_num <= total_gpu_num                                                                                                                                                                      
    if max_process_per_gpu > 1:                                                                                                                                                                                    
        warnings.warn("max_process_per_gpu>=2 may get stuck on some computers or clusters.")                                                                                                                       
                                                                                                                                                                                                                   
    # initialize input and model                                                                                                                                                                                   
    total_input_num = 10                                                                                                                                                                                           
    input = [torch.ones(1, 1, 2, 2)] * total_input_num                                                                                                                                                             
    output = []                                                                                                                                                                                                    
                                                                                                                                                                                                                   
    def mycallback(arg):                                                                                                                                                                                           
        output.append(arg[0])                                                                                                                                                                                      
                                                                                                                                                                                                                   
    N = 5000002                                                                                                                                                                                                    
    keep_per_sample = 512                                                                                                                                                                                          
    quantile = 1 - float(keep_per_sample/N)                                                                                                                                                                        
    print(quantile)                                                                                                                                                                                                
                                                                                                                                                                                                                   
    start_time = timeit.default_timer()     
    
    # (2) Estimate quantile
    chunk_size, num_samples = 10, N
    tgt_train_full = torch.nn.functional.normalize(torch.randn(N, 768))
    src_train_full = []
    # randomly initialize A
    for i in range(ngpus):
        # each GPU has a slice of A
        slice = torch.nn.functional.normalize(torch.randn(N // ngpus, 768))
        src_train_full.append(slice)

    # test the model on multiple GPUs distributedly
    pool = multiprocessing.Pool(available_gpu_num * max_process_per_gpu)
    for i in range(7):
        # hint: pool.apply_async cannot output informative debugging logs. Use pool.apply() for debugging.
        pool.apply_async(multi_gpu_testing_wrapper, args=(src_train_full[i], tgt_train_full, quantile, i,  None, available_gpu_num), callback=mycallback)
        #pool.apply(multi_gpu_testing_wrapper, args=(src_train_full[i], tgt_train_full_[i], i,  0, available_gpu_num))

    pool.close()
    pool.join()
    print('All subprocesses done.')
    
    quantiles = list(itertools.chain.from_iterable(output))
    
    # (3) Get similarity graph thresholded on quantile                                                                                                                                                           
    quantile = statistics.median(quantiles)
    
    output = []
    def mycallback(arg):
        output.append(arg[0])
        
    pool = multiprocessing.Pool(available_gpu_num * max_process_per_gpu)
    for i in range(7):
        # hint: pool.apply_async cannot output informative debugging logs. Use pool.apply() for debugging.
        pool.apply_async(multi_gpu_testing_wrapper2, args=(src_train_full[i], tgt_train_full, quantile, i,  None, available_gpu_num), callback=mycallback)
        #pool.apply(multi_gpu_testing_wrapper, args=(src_train_full[i], tgt_train_full_[i], i,  0, available_gpu_num))

    pool.close()
    pool.join()
    
    row = list(itertools.chain.from_iterable([z[0] for z in output]))
    col = list(itertools.chain.from_iterable([z[1] for z in output]))
    data = list(itertools.chain.from_iterable([z[2] for z in output]))

    # (4) Get permutation using graph bandwidth minimization on sparsified graph (cuthill-mckee)                                                                                                                 
    permutation = list(reverse_cuthill_mckee(csr_matrix((data, (row, col)),                                                                                                                                      
                                                shape=(num_samples, num_samples)))) 

    print(timeit.default_timer() - start_time)

   # # check output quality (sometimes a process may fail due to out-of-momory error, but there is no error!)
   #  print(f'\n{len(output)}/{len(src_train_full)} processes succeeded.')
   #  assert len(src_train_full) == len(output)

if __name__ == '__main__':
    main()