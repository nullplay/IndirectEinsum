import torch
import sys
import os

from indirecteinsum import Insum 
from torch.sparse._triton_ops_meta import optimize_bsr_dense_addmm

import warnings
from torch._inductor.runtime.benchmarking import benchmarker

def group_and_pad(w, o, val,window_size):
    device = "cuda"
    w_max = w.max()
    o_max = o.max()
    w_multiplier = (o_max + 1)
    key = w * w_multiplier + o 

    # Perform the sort
    _, sort_indices = torch.sort(key)
    
    w_sorted = w[sort_indices]
    o_sorted = o[sort_indices]
    val_sorted = val[sort_indices]
    
    unique_w, counts = torch.unique_consecutive(w_sorted, return_counts=True)
    counts = counts.long()  # Ensure counts are integers
    padded_counts = ((counts.float() / window_size).ceil().long()) * window_size
    positions = torch.cat([torch.tensor([0], device=device), torch.cumsum(padded_counts[:-1], dim=0)])
    total_padded_length = padded_counts.sum()
    w_padded = unique_w.repeat_interleave(padded_counts)
    o_padded = torch.zeros(total_padded_length, dtype=o.dtype, device=device)
    val_padded = torch.zeros(total_padded_length, dtype=val.dtype, device=device)
    total_counts = counts.sum()
    starts = torch.cumsum(counts, dim=0) - counts  # Start indices for each group in the sorted arrays
    starts_repeated = starts.repeat_interleave(counts)
    offsets_within_group = torch.arange(total_counts, device=device) - starts_repeated
    positions_repeated = positions.repeat_interleave(counts)
    indices = positions_repeated + offsets_within_group

    o_padded[indices] = o_sorted
    val_padded[indices] = val_sorted

    res1 = w_padded[::window_size].contiguous() #P0
    res2 = o_padded.reshape(-1,window_size) #P0xP1
    res3 = val_padded.reshape(-1,window_size) #P0xP1

    return res1, res2, res3

def group_and_pad_spmm(w, o, val,window_size):
    device = "cuda"
    w_max = w.max()
    o_max = o.max()
    w_multiplier = (o_max + 1)
    key = w * w_multiplier + o 

    # Perform the sort
    _, sort_indices = torch.sort(key)
    
    w_sorted = w[sort_indices]
    o_sorted = o[sort_indices]
    val_sorted = val[sort_indices,:]
    
    unique_w, counts = torch.unique_consecutive(w_sorted, return_counts=True)
    counts = counts.long()  # Ensure counts are integers
    padded_counts = ((counts.float() / window_size).ceil().long()) * window_size
    positions = torch.cat([torch.tensor([0], device=device), torch.cumsum(padded_counts[:-1], dim=0)])
    total_padded_length = padded_counts.sum()
    w_padded = unique_w.repeat_interleave(padded_counts)
    o_padded = torch.zeros(total_padded_length, dtype=o.dtype, device=device)
    val_padded = torch.zeros((total_padded_length,val.shape[1]), dtype=val.dtype, device=device)
    total_counts = counts.sum()
    starts = torch.cumsum(counts, dim=0) - counts  # Start indices for each group in the sorted arrays
    starts_repeated = starts.repeat_interleave(counts)
    offsets_within_group = torch.arange(total_counts, device=device) - starts_repeated
    positions_repeated = positions.repeat_interleave(counts)
    indices = positions_repeated + offsets_within_group

    o_padded[indices] = o_sorted
    val_padded[indices,:] = val_sorted

    res1 = w_padded[::window_size].contiguous() #P0
    res2 = o_padded.reshape(-1,window_size) #P0xP1
    res3 = val_padded.reshape(-1,window_size,val.shape[1]) #P0xP1xbM

    return res1, res2, res3


warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision("high")
torch._inductor.config.triton.prefer_nd_tiling=True
torch._inductor.config.triton.native_matmul=True
Insum = torch.compile(Insum)
bench = benchmarker.benchmark_gpu


# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
M, K, N = 1024, 1024, 1024   # Full dense dimensions.
blocksize = (32, 32)         # Block size for the BSR format.
sparsities = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]  # Different sparsity levels
dtypes = [torch.float16]  # Data types to test

def create_bsr_matrix(M, K, blocksize, sparsity, dtype):
    assert M % blocksize[0] == 0 and K % blocksize[1] == 0
    num_blocks_M = M // blocksize[0]
    num_blocks_K = K // blocksize[1]
    mask = torch.bernoulli(torch.full((num_blocks_M, num_blocks_K), 1 - sparsity, dtype=dtype))
    mask_full = mask.repeat_interleave(blocksize[0], dim=0).repeat_interleave(blocksize[1], dim=1)
    dense = torch.randn(M, K, dtype=dtype, device=torch.device("cuda")) * mask_full * 0.1
    return dense.to_sparse_bsr(blocksize)

for dtype in dtypes:
    print(f"Testing dtype: {dtype}")
    for sparsity in sparsities:
        print(f"  Sparsity: {sparsity}")

        A = create_bsr_matrix(M, K, blocksize, sparsity, dtype)
        Acsr = A.to_dense().to_sparse_csr()
        B = torch.randn(K, N, dtype=dtype, device=A.device) * 0.1

        # Extract COO representation
        COO = A.to_sparse().coalesce().indices()
        CooI, CooK = COO[0].clone(), COO[1].clone()
        CooV = A.to_sparse().coalesce().values().clone()
        CooI2, CooK2, CooV2 = group_and_pad(CooI, CooK, CooV, 16)
        CooI2, CooK2, CooV2 = CooI2.cuda().contiguous(), CooK2.cuda().contiguous(), CooV2.cuda().contiguous()

        # Extract block COO representation
        crow_indices = A.crow_indices()
        Ai = torch.arange(len(crow_indices) - 1, device=A.device).repeat_interleave(crow_indices.diff()).clone()
        Ak = A.col_indices().detach().clone()
        Av = A.values().detach().clone().reshape(-1, blocksize[0]*blocksize[1])
        x,y = torch.unique(Ai, return_counts=True)

        window_size_star = torch.sqrt(torch.tensor( y.sum() / (y.shape[0])))

        def near_powers_of_two(x: float) -> tuple[int, int]:
            if x <= 1:
                return 2, 4
            import math
            exp = math.log2(x)
            exp_floor = math.floor(exp)
            exp_ceil = math.ceil(exp)
            lower = 2 ** exp_floor
            upper = 2 ** exp_ceil
            if lower == upper:
                return 2 ** (exp_floor - 1), lower
            return lower, upper

        
        print("    Dense mm performance:")
        A_dense = A.to_dense()
        ref = torch.matmul(A_dense, B)
        print(bench(lambda: torch.matmul(A_dense, B)))

        C = torch.zeros((M,N), device=B.device, dtype=B.dtype)
        print("    Custom coo spmm performance:")
        B = B.view((K, N)).contiguous()
        Insum("C[Ai[p],n] += Av[p] * B[Ak[p],n]", C=C.zero_(), Av=CooV, Ai=CooI, Ak=CooK, B=B)
        print(bench(lambda: Insum("C[Ai[p],n] += Av[p] * B[Ak[p],n]", C=C.zero_(), Av=CooV, Ai=CooI, Ak=CooK, B=B)))
        torch.testing.assert_close(ref, C, atol=1e-2, rtol=1e-2)

        print("    Custom groupcoo spmm performance:")
        Insum("C[Ai[g],n] += Av[g,p] * B[Ak[g,p],n]", C=C.zero_(), Av=CooV2, Ai=CooI2, Ak=CooK2, B=B)
        print(bench(lambda: Insum("C[Ai[g],n] += Av[g,p] * B[Ak[g,p],n]", C=C.zero_(), Av=CooV2, Ai=CooI2, Ak=CooK2, B=B)))
        torch.testing.assert_close(ref, C, atol=1e-2, rtol=1e-2)


        for window_size in near_powers_of_two(window_size_star):
            Ai2, Ak2, Av2 = group_and_pad_spmm(Ai, Ak, Av, window_size)
            Av2 = Av2.reshape(-1,window_size,blocksize[0], blocksize[1])
            Ai2, Ak2, Av2 = Ai2.cuda().contiguous(), Ak2.cuda().contiguous(), Av2.cuda().contiguous()
            Av2 = Av2.permute(0, 2, 3, 1).contiguous()

            B = B.view((K // blocksize[1], blocksize[1], N)).contiguous()
            print("    ours performance (group_size: ", window_size, ") " )
            C = torch.zeros((M // blocksize[0], blocksize[0], N), dtype=torch.float16, device=B.device)
            out = Insum("C[Ai[g],m,n] += Av[g,m,k,p] * B[Ak[g,p],k,n]", C=C.zero_(), Av=Av2, Ai=Ai2, Ak=Ak2, B=B)
            print(bench(lambda: Insum("C[Ai[g],m,n] += Av[g,m,k,p] * B[Ak[g,p],k,n]", C=C.zero_(), Av=Av2, Ai=Ai2, Ak=Ak2, B=B)))
            torch.testing.assert_close(ref, C.reshape(M,N), atol=1e-2, rtol=1e-2)


        B = B.view((K, N)).contiguous()
        print("    torchbsr performance:")
        from torch.sparse._triton_ops import bsr_dense_mm
        ref = bsr_dense_mm(A, B)
        print(bench(lambda: bsr_dense_mm(A, B)))
        torch.testing.assert_close(ref, out.reshape(M,N), atol=1e-2, rtol=1e-2)


