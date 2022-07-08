# JAX Triton integration


## Installation

You may optionally use a `virtualenv` or can use `pip install --user`.

1. Install latest Triton

```bash
$ git clone https://github.com/openai/triton.git
$ cd triton/python
$ pip install cmake
$ pip install -e .[tests]
```

To verify it worked, try running (from within `triton/python`):
```bash
$ pytest test/unit
```

2. Get JAX w/ Triton support

```bash
$ git clone https://github.com/sharadmv/jax.git
$ cd jax
$ git checkout triton
$ pip install -e ".[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
$ pip install pybind11
$ cd triton
$ make # compiles our custom call
```

We have a couple examples already written. Try running (inside of `jax/triton/examples`):
```bash
$ python matrix_multiplication.py
```
