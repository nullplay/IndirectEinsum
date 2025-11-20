# IndirectEinsum

In order to get the maximum performance on GPU, you will need PyTorch with native_matmul enabled.
As of Nov 2025, it is available in PyTorch nightly. (https://pytorch.org/get-started/locally/)

To run the **SpMM COO** example in `example.py`, use:

```bash
python example.py
```

If you want to inspect the FX graph and view the generated Triton code, run:

```bash
TORCH_LOGS="post_grad_graphs,output_code" python example.py
```

