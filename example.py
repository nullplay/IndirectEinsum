import torch
from indirecteinsum import Insum 

torch.set_default_device("cuda")

M, K = 64, 64
nnz = (M * K) // 2   # 50% of 4096 = 2048

# choose random nonzero coords and values 
lin_idx = torch.randperm(M * K)[:nnz]
rows = lin_idx // M
cols = lin_idx %  K
values = torch.randn(nnz)

# Dense output and input
N = 16
out = torch.zeros((M,N))
In = torch.rand((K,N))

@torch.compile
def SpMM_COO(out, In, rows, cols, values) :
    # Note that Insum always inplace update the left hand side (out) 
    out = Insum("C[Row[p],n] += Val[p] * B[Col[p],n]",
        C = out,
        B = In,
        Row = rows,
        Col = cols,
        Val = values,
    )
    return out

# Run Indirect Einsum with torch.compile
out = SpMM_COO(out, In, rows, cols, values)

# dense version for correctness 
A_dense = torch.zeros(M, K)
A_dense[rows, cols] = values

torch.testing.assert_close(A_dense @ In, out)
