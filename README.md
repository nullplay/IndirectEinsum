# IndirectEinsum

To run the **SpMM COO** example in `example.py`, use:

```bash
python example.py
```

If you want to inspect the FX graph and view the generated Triton code, run:

```bash
TORCH_LOGS="post_grad_graphs,output_code" python example.py
```

