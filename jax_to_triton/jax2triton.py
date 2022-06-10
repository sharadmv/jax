import torch

from typing import Any, Callable, Tuple, Union
from types import SimpleNamespace

import jax
import jax.dlpack
from jax import core
import jax.numpy as jnp
import triton
import triton.language as tl
from jax.lib import xla_client as xc
from jax.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
import numpy as np

import custom_call

xc.register_custom_call_target("triton_call", custom_call.get_custom_call(), platform="CUDA")

def get_triton_type(obj: Any) -> str:
    type_map = {
        jnp.dtype("float32"): "f32",
        jnp.dtype("int32"): "i32",
    }
    if isinstance(obj, jax.core.ShapedArray):
        return type_map[obj.dtype]
    if isinstance(obj, tl.constexpr):
        obj = obj.value
    if isinstance(obj, int):
        if -2**31 <= obj < 2**31:
            return 'i32'
        elif 2**31 <= obj < 2**32:
            return 'u32'
        elif -2**63 <= obj < 2**63:
            return 'i64'
        elif 2**63 <= obj < 2**64:
            return 'u64'
        else:
            raise ValueError(f'integer overflow representing {obj}')
    if isinstance(obj, float):
        return 'f'
    if isinstance(obj, bool):
        return 'B'
    if isinstance(obj, str):
        return 'str'
    raise NotImplementedError(f'could not compute type name for {obj}')
    
def get_triton_python_ir(aval):
    if aval.shape == ():
        return "scalar", get_triton_type(aval)
    return "ptr", get_triton_type(aval)

def compile(triton_function, constants, *, key, device=0):
    def lower(*args):
        arg_types = [get_triton_python_ir(a) for a in args]
        attributes = {i: 16 for i in range(len(args))}
        triton_function._warmup(arg_types=arg_types, device=device, attributes=attributes, constants=constants, num_warps=4, num_stages=2, key=key, is_manual_warmup=True)
        pass
    return lower

def j2t(x_jax):
  x_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x_jax))
  return x_torch

def t2j(x_torch):
  x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
  x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch))
  return x_jax

Metaparameters = Any
ShapeDtypeDuck = Any

triton_call_p = jax.core.Primitive('triton_call')
triton_call = triton_call_p.bind

table = {'float32': torch.float32, 'int32': torch.int32}

@triton_call_p.def_impl
def triton_call_impl(*args, kernel, out_shape, grid, **metaparams):
  args_torch = [j2t(x) for x in args]
  output_torch = torch.empty(out_shape.shape, dtype=table[out_shape.dtype.name],
                             device=torch.device('cuda:0'))
  kernel[grid](*args_torch, output_torch, **metaparams)
  return t2j(output_torch)

@triton_call_p.def_abstract_eval
def triton_call_abstract_eval(*_, out_shape, **__):
  return core.ShapedArray(out_shape.shape, out_shape.dtype)

def avals_to_layouts(avals):
  return ir.ArrayAttr.get([aval_to_layout(a) for a in avals])

def aval_to_layout(aval):
  arange = np.arange(aval.ndim, dtype='int64')[::-1].copy()
  return ir.DenseIntElementsAttr.get(arange, type=ir.IndexType.get())

def emit_triton_call(triton_func, avals_in, avals_out, **metaparams):
  aval_out, = avals_out
  metadata = {triton_func.arg_names.index(k) : v for k, v in metaparams.items()}
  # import inspect
  # breakpoint()
  compile(triton_func, metadata, key="foo")(*avals_in, aval_out)
  loaded_binary = triton_func.bin_cache["foo"]
  kernel_ptr = loaded_binary.kernel
  shared_mem = loaded_binary.shared_mem
  grid_0, grid_1, grid_2 = 1, 1, 1
  arity = len(avals_in) + 1
  descriptor = custom_call.make_triton_call_descriptor(kernel_ptr, shared_mem, grid_0, grid_1, grid_2, arity)
  return descriptor

def triton_call_lowering(ctx, *args, kernel, out_shape, grid, **metaparams):
  out_type = ir.RankedTensorType.get(out_shape.shape, mlir.dtype_to_ir_type(out_shape.dtype))
  i32_type = ir.IntegerType.get_signless(32)
  descriptor = emit_triton_call(kernel, ctx.avals_in, ctx.avals_out, **metaparams)
  n_elems = ctx.avals_out[0].size
  out = mhlo.CustomCallOp(
            [out_type], args,
            call_target_name=ir.StringAttr.get("triton_call"),
            has_side_effect=ir.BoolAttr.get(False),
            backend_config=ir.StringAttr.get(descriptor),
            api_version=ir.IntegerAttr.get(i32_type, 1),
            called_computations=ir.ArrayAttr.get([]),
            # operand_layouts=avals_to_layouts(ctx.avals_in + [core.ShapedArray((), jnp.int32)]),
            operand_layouts=avals_to_layouts(ctx.avals_in),
            result_layouts=avals_to_layouts(ctx.avals_out))
  return out.results
mlir.register_lowering(triton_call_p, triton_call_lowering)
