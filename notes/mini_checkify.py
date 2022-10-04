from __future__ import annotations
import dataclasses
import functools

from typing import Any, Dict, FrozenSet, Type

import jax
import jax.numpy as jnp
from jax import core
from jax import lax
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node_class, tree_leaves
from jax.api_util import flatten_fun
from jax import linear_util as lu
from jax._src.util import safe_map, safe_zip, split_list
from jax._src import source_info_util

import numpy as np

jax.config.update("jax_traceback_filtering", "off")

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

class SimpleRepr(type):
  def __repr__(cls):
    return cls.__name__

class JaxError(Exception, metaclass=SimpleRepr):
  pass

class DivideByZero(JaxError):

  def __str__(self):
    return "Divide by zero!"

  def __repr__(self):
    return "DivideByZero"

class AssertionError(JaxError):
  def __init__(self, fmt, *format_args, **format_kwargs):
    self.fmt = fmt
    self.format_args = format_args
    self.format_kwargs = format_kwargs

  def __str__(self):
    return self.fmt.format(*self.format_args, **self.format_kwargs)

  def __repr__(self):
    return "AssertionError"

@register_pytree_node_class
@dataclasses.dataclass
class Error:
  pred: Dict[Type[JaxError], bool]
  code: Dict[Type[JaxError], Any]
  ctors: Dict[int, Any]
  _counter: int

  def tree_flatten(self):
    ctor_keys = sorted(self.ctors.keys())
    ctor_funcs = [self.ctors[k][0] for k in ctor_keys]
    ctor_vals = [self.ctors[k][1] for k in ctor_keys]
    return (ctor_vals, list(self.pred.values()), list(self.code.values())), (self.pred.keys(),
        self.code.keys(), ctor_funcs, ctor_keys, self._counter)

  def has_error(self):
    return jnp.any(jnp.asarray(list(self.pred.values())))

  @classmethod
  def tree_unflatten(cls, data, xs):
    ctor_vals, pred_vals, code_vals = xs
    pred_keys, code_keys, ctor_funcs, ctor_keys, counter = data
    ctors = dict(zip(ctor_keys, zip(ctor_funcs, ctor_vals)))
    return Error(dict(zip(pred_keys, pred_vals)), dict(zip(code_keys,
      code_vals)), ctors, counter)

  def update(self, error_type, pred, ctor, *args):
    curr_pred = self.pred[error_type]
    curr_code = self.code[error_type]
    next_pred = curr_pred | pred
    next_code = lax.select(curr_pred, curr_code, self._counter)
    return Error(
        self.pred | {error_type: next_pred},
        self.code | {error_type: next_code},
        self.ctors | {self._counter: (ctor, args)},
        self._counter + 1)

  def maybe_raise(self):
    for err_type, code in sorted(self.code.items(), key=lambda x: x[1]):
      if self.pred[err_type]:
        ctor, args = self.ctors[int(code)]
        raise ctor(*args)

def init_error(errors: FrozenSet[Type[JaxError]]) -> Error:
  return Error({err: False for err in errors}, {err: -1 for err in errors}, {}, 0)

discharge_rules = {}

class DischargeTrace(core.Trace):
  pure = lift = lambda self, val: CheckifyTracer(self, val)

  def __init__(self, main: core.MainTrace, sublevel: core.Sublevel,
      enabled_errors: FrozenSet[Type[JaxError]]) -> None:
    self.main = main
    self.level = main.level
    self.sublevel = sublevel
    self.enabled_errors = enabled_errors

  def sublift(self, tracer):
    return CheckifyTracer(self, tracer.val)

  def process_primitive(self, primitive, tracers, params):
    in_vals = [t.val for t in tracers]
    in_avals = [t.aval for t in tracers]
    _, effects = primitive.abstract_eval(*in_avals, **params)
    rule = discharge_rules.get(primitive)
    if any(err in effects for err in self.enabled_errors):
      rule = discharge_rules.get(primitive, None)
      if not rule:
        raise NotImplementedError(f"Discharge rule not implemented for: {primitive}")
      self.main.error, out = rule(self.main.error, self.enabled_errors,
                                  *in_vals, **params)
    else:
      out = primitive.bind(*in_vals, **params)
    if primitive.multiple_results:
      return [CheckifyTracer(self, x) for x in out]
    else:
      return CheckifyTracer(self, out)

  def process_call(self, call_primitive, f, tracers, params):
    in_vals = [t.val for t in tracers]
    in_vals = [t.val for t in tracers]
    e = self.main.error
    del self.main.error
    flat_error, error_tree = tree_flatten(e)
    f, error_thunk = checkify_subtrace(
        f, error_tree, len(flat_error), self.main)
    if 'donated_invars' in params:
      params = dict(params, donated_invars=(*((False,) * len(flat_error)),
                                            *params['donated_invars']))
    error_and_out_vals = call_primitive.bind(f, *flat_error, *in_vals, **params)
    num_error_vals, error_tree = error_thunk()
    error_vals, out_vals = split_list(error_and_out_vals, [num_error_vals])
    error = tree_unflatten(error_tree, error_vals)
    self.main.error = error
    return [CheckifyTracer(self, x) for x in out_vals]

class CheckifyTracer(core.Tracer):
  def __init__(self, trace, val):
    self._trace = trace
    self.val = val
  aval = property(lambda self: core.get_aval(self.val))
  full_lower = lambda self: self

def checkify_flat(fun: lu.WrappedFun, enabled_errors: FrozenSet[Type[JaxError]],
                  *args):
  error = init_error(enabled_errors)
  flat_error, error_tree = tree_flatten(error)
  fun, error_thunk = checkify_subtrace(fun, error_tree, len(flat_error))
  fun = checkify_traceable(fun, enabled_errors)
  outvals = fun.call_wrapped(*flat_error, *args)
  num_error_vals, error_tree = error_thunk()
  error_vals, outvals = split_list(outvals, [num_error_vals])
  error = tree_unflatten(error_tree, error_vals)
  return (error, outvals)

@lu.transformation
def checkify_traceable(enabled_errors, *args):
  with core.new_main(DischargeTrace, enabled_errors=enabled_errors) as main:
    outs = yield (main, *args), {}
    del main
  yield outs

@lu.transformation_with_aux
def checkify_subtrace(error_tree, error_size, main, *error_and_args):
  error_vals, args = split_list(error_and_args, [error_size]) 
  error = tree_unflatten(error_tree, error_vals)
  main.error = error
  trace = main.with_cur_sublevel()
  in_tracers = [CheckifyTracer(trace, x) for x in args]
  out = yield in_tracers, {}
  out_tracers = map(trace.full_raise, out)
  out_vals = [t.val for t in out_tracers]
  error = main.error
  del main.error
  flat_error, error_tree = tree_flatten(error)
  yield (*flat_error, *out_vals), (len(flat_error), error_tree)

def discharge_errors(fun, errors):
  def checked_fun(*args, **kwargs):
    args_flat, in_tree = tree_flatten((args, kwargs))
    f, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
    error, out_flat = checkify_flat(f, errors, *args_flat)
    out = tree_unflatten(out_tree(), out_flat)
    return error, out
  return checked_fun


raise_p = core.Primitive("raise")
raise_p.multiple_results = True

def _raise_impl(*args, tree):
  error = tree_unflatten(tree, args)
  error.maybe_raise()
  return []
raise_p.def_impl(_raise_impl)

def _raise_abstract_eval(*args, tree):
  error = tree_unflatten(tree, args)
  return [], set(error.pred.keys())
raise_p.def_effectful_abstract_eval(_raise_abstract_eval)

def raise_(error):
  err, tree = tree_flatten(error)
  raise_p.bind(*err, tree=tree)

def _raise_discharge_rule(outer_error, error_types, *args, tree):
  inner_error = tree_unflatten(tree, args)
  new_preds = {}
  new_codes = {}
  for error_type in error_types:
    if error_type in inner_error.pred:
      outer_pred = outer_error.pred[error_type]
      inner_pred = inner_error.pred[error_type]
      outer_code = outer_error.code[error_type]
      inner_code = inner_error.code[error_type]
      new_preds[error_type] = outer_pred | inner_pred
      new_codes[error_type] = lax.select(outer_pred, outer_code, inner_code)
  new_ctors = {code + outer_error._counter: ctor for code, ctor in
      inner_error.ctors.items()}
  new_error = Error(new_preds, new_codes, outer_error.ctors | new_ctors,
      outer_error._counter + len(new_ctors))
  new_preds = {}
  new_codes = {}
  for error_type in inner_error.pred:
    if error_type in error_types:
      continue
    new_preds[error_type] = inner_error.pred[error_type]
    new_codes[error_type] = inner_error.code[error_type]
  raise_(Error(new_preds, new_codes, inner_error.ctors, inner_error._counter))
  return new_error, []
discharge_rules[raise_p] = _raise_discharge_rule

div_p = core.Primitive("div")

def _div_impl(x, y, should_check: bool):
  if should_check and np.array(y == 0).any():
    raise DivideByZero()
  return lax.div(x, y)
div_p.def_impl(_div_impl)

def _div_abstract_eval(x, y, should_check: bool):
  if should_check:
    return x, {DivideByZero}
  return x, set()
div_p.def_effectful_abstract_eval(_div_abstract_eval)

def _div_error_discharge_rule(error: Error, error_types, x, y, *, should_check: bool):
  divide_by_zero = y == 0
  def _factory():
    return DivideByZero()
  error = error.update(DivideByZero, divide_by_zero, _factory)
  return error, lax.div(x, y)

discharge_rules[div_p] = _div_error_discharge_rule
div = functools.partial(div_p.bind, should_check=True)

assert_p = core.Primitive("assert")
assert_p.multiple_results = True

def assert_(pred, fmt, *args, **kwargs):
  flat_args, tree = tree_flatten((args, kwargs))
  return assert_p.bind(pred, *flat_args, tree=tree, fmt=fmt)

def _assert_impl(pred, *args, fmt, tree):
  if not pred:
    format_args, format_kwargs = tree_unflatten(tree, args)
    raise AssertionError(fmt, *format_args, **format_kwargs)
  return []
assert_p.def_impl(_assert_impl)

def _assert_abstract_eval(pred, *args, fmt, tree):
  return [], {AssertionError}
assert_p.def_effectful_abstract_eval(_assert_abstract_eval)

def _assert_discharge_rule(error, error_types, pred, *args, fmt, tree):
  def _factory(*args):
    format_args, format_kwargs = tree_unflatten(tree, args)
    return AssertionError(fmt, *format_args, **format_kwargs)
  error = error.update(AssertionError, ~pred, _factory, *args)
  return error, []
discharge_rules[assert_p] = _assert_discharge_rule

def try_except_then(f, error_types, handler, *args, **kwargs):
  error, out = discharge_errors(f, error_types)(*args, **kwargs)
  return lax.cond(error.has_error(), lambda: handler(*args, **kwargs),
                  lambda: out)
                  

def f(x, y):
  z = div(x, y)
  assert_(z != 0., "Zero result: {x}, {y}", x=x, y=y)
  assert_(z != 1., "One result: {x}, {y}", x=x, y=y)
  return z

@jax.jit
def g(x, y):
  return try_except_then(f, {DivideByZero}, lambda x, y: -1., x, y)

print(jax.make_jaxpr(g)(1., 1.).effects)  # AssertionError
error, out = discharge_errors(g, {AssertionError})(1., 1.)
raise_(error)  # Raises assertion
