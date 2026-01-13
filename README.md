# IndirectEinsum


`Insum` executes an **indirect einsum (gather–scatter tensor algebra)** using `torch.compile`, allowing index indirection (e.g. `Row[p]`, `Col[p]`) and accumulation semantics (`+=`) to be compiled into efficient kernels.

**Example:**  
```python
out = Insum("C[Row[p],n] += Val[p] * B[Col[p],n]",
            C=out, B=In, Row=rows, Col=cols, Val=values)
```


In order to get the maximum performance on GPU, you will need PyTorch with `native_matmul` enabled.
`native_matmul` is released as PyTorch 2.10.  

To run the **SpMM COO** example in `example.py`, use:

```bash
python example.py
```

If you want to inspect the FX graph and view the generated Triton code, run:

```bash
TORCH_LOGS="post_grad_graphs,output_code" python example.py
```

