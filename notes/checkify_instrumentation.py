from __future__ import annotations

from typing import Any

import jax
from jax import core
from jax import lax
from jax._src.lax.lax import _unbroadcast
from jax.interpreters import ad
from jax._src import ad_util
from jax._src.lax import control_flow as cf
import jax.numpy as jnp
import enum
import contextlib


class Check(enum.Enum):
  DIV_BY_ZERO = enum.auto()
  INVALID_NAN = enum.auto()

cf.allowed_effects.add(Check.DIV_BY_ZERO)

_active_checks: list[frozenset[Check]] = [frozenset()]

@contextlib.contextmanager
def instrument(*checks: Check):
  _active_checks.append(frozenset(checks))
  yield
  _active_checks.pop()

def _current_checks():
  return _active_checks[-1]

div_p = core.Primitive('div')

@div_p.def_impl
def _div_impl(x, y, *, check):
  if check and y == 0: raise ZeroDivisionError("division by zero")
  return jnp.divide(x, y)

@div_p.def_effectful_abstract_eval
def _div_abstract_eval(x, y, *, check):
  if check:
    return x, {Check.DIV_BY_ZERO}
  return x, set()

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
  should_check = Check.DIV_BY_ZERO in _current_checks()
  return div_p.bind(x, y, check=should_check)

@jax.grad
def f(x, y):
  z = div(x, y)
  with instrument(Check.DIV_BY_ZERO):
    def body(i, x):
      x = div(x, y)
      return x
    return lax.fori_loop(0, 5, body, z)

print("f(2., 1.)", f(2., 1.))
print("f(1., 0.)", f(1., 0.))

jaxpr = jax.make_jaxpr(f)(1., 0.).jaxpr
print(jaxpr)
print(jaxpr.effects)
print(jax.make_jaxpr(lambda x, y: core.eval_jaxpr(jaxpr, (), x, y))(1, 0))
