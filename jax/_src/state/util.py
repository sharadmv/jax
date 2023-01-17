# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for state utilities."""
from typing import Any, List, Sequence, Tuple

from jax import core
from jax import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax._src.api_util import flatten_fun_nokwargs, 
from jax._src.state import types as state_types
from jax._src.state.primitives import ref_get
from jax._src.tree_util import PyTreeDef, treedef_tuple
from jax._src.util import split_list, merge_lists, partition_list

ShapedArrayRef = state_types.ShapedArrayRef

### Tracing utilities

def hoist_consts_to_refs(jaxpr: core.Jaxpr) -> core.Jaxpr:
  all_const_avals = [var.aval for var in jaxpr.constvars]
  is_const_ref = [isinstance(var.aval, ShapedArrayRef) for var in
                  jaxpr.constvars]
  const_avals, const_ref_avals = partition_list(is_const_ref, all_const_avals)
  const_avals = [ShapedArrayRef(aval.shape, aval.dtype) for aval in const_avals]  # pytype: disable=attribute-error
  merged_const_avals = merge_lists(is_const_ref, const_avals, const_ref_avals)
  arg_avals = [var.aval for var in jaxpr.invars]
  in_avals = [*merged_const_avals, *arg_avals]
  num_consts = len(merged_const_avals)

  def _hoist(*consts_args):
    all_consts, args = split_list(consts_args, [num_consts])
    consts, const_refs = partition_list(is_const_ref, all_consts)
    # We immediately read the const values out of the `Ref`s.
    consts = map(lambda x: ref_get(x, ()), consts)
    all_consts = merge_lists(is_const_ref, consts, const_refs)
    return core.eval_jaxpr(jaxpr, all_consts, *args)
  hoisted_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(_hoist), in_avals)
  assert not consts, "All consts should have been converted to refs"
  return hoisted_jaxpr

def val_to_ref_aval(x: Any) -> ShapedArrayRef:
  aval = core.raise_to_shaped(core.get_aval(x))
  if type(aval) is not core.ShapedArray:
    raise Exception(f"can't make ref from {x}")
  return ShapedArrayRef(aval.shape, aval.dtype)

