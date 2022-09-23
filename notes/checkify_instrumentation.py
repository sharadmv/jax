from __future__ import annotations
from functools import partial

from typing import Any

import jax
from jax import core
from jax import lax
from jax._src.lax.lax import _unbroadcast, _nary_lower_mhlo
from jax.interpreters import ad
from jax.interpreters import mlir
from jax._src.lib.mlir.dialects import mhlo
from jax._src import ad_util
from jax._src import checkify
from jax._src.lax import control_flow as cf
import jax.numpy as jnp
import enum
import contextlib


_active_checks: list[frozenset[checkify.ErrorCategory]] = [frozenset()]

@contextlib.contextmanager
def instrument(*checks: checkify.ErrorCategory):
  _active_checks.append(frozenset(checks))
  yield
  _active_checks.pop()

def _current_checks():
  return _active_checks[-1]

div_p = core.Primitive('div')

@div_p.def_impl
def _div_impl(x, y, *, check):
  if check:
    # Could do effectful lowering
    raise ValueError("Cannot execute effectful `div`.")
  return jnp.divide(x, y)

@div_p.def_effectful_abstract_eval
def _div_abstract_eval(x, y, *, check):
  if check:
    return x, {checkify.ErrorCategory.DIV}
  return x, set()

def _div_error_check(error, error_categories, x, y, *, check):
  if not check:
    return div_p.bind(x, y, check=False), error
  if checkify.ErrorCategory.DIV not in error_categories:
    return div_p.bind(x, y, check=True), error
  any_zero = jnp.any(jnp.equal(y, 0))
  msg = f'divided by zero at {checkify.summary()}'
  error = checkify.assert_func(error, any_zero, msg, None)
  return div_p.bind(x, y, check=False), error
checkify.error_checks[div_p] = _div_error_check

def _div_lowering(ctx, x, y, *, check):
  if check:
    # Could do effectful lowering
    raise ValueError("Cannot execute effectful `div`.")
  return _nary_lower_mhlo(mhlo.DivOp, ctx, x, y)
mlir.register_lowering(div_p, _div_lowering)

ad.defjvp(div_p,
          lambda g, x, y, check: div_p.bind(g, y, check=check),
          lambda g, x, y, _: lax.mul(lax.mul(lax.neg(g), x), lax.integer_pow(y, -2)))

def _div_transpose_rule(cotangent, x, y, *, check):
  assert ad.is_undefined_primal(x) and not ad.is_undefined_primal(y)
  if type(cotangent) is ad_util.Zero:
    return [ad_util.Zero(x.aval), None]
  else:
    return [_unbroadcast(x.aval, div_p.bind(cotangent, y, check=check)), None]
ad.primitive_transposes[div_p] = _div_transpose_rule

def div(x, y):
  should_check = checkify.ErrorCategory.DIV in _current_checks()
  if should_check:
    x = jax.core.raise_as_much_as_possible(x)
    y = jax.core.raise_as_much_as_possible(y)
  return div_p.bind(x, y, check=should_check)

@jax.jit
def f(x, y):
  def g(z, y):
    with instrument(checkify.ErrorCategory.DIV):
      return div(z, y)
  # Try catch?
  err, out = checkify.checkify(g, checkify.div_checks)(z, y)
  return lax.cond(err.err, lambda: jnp.nan, lambda: out)

jaxpr = jax.make_jaxpr(f)(1., 0.).jaxpr
print(jaxpr, jaxpr.effects)

checkify_jaxpr = jax.make_jaxpr(checkify.checkify(f, checkify.all_checks))(2., 0.).jaxpr
print(checkify_jaxpr)

print(f(1., 2.)) # 0.5
print(f(1., 0.)) # nan
