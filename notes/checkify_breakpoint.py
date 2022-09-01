from jax.experimental import checkify
import jax.numpy as jnp
from jax import lax
import jax
from jax.config import config
config.update('jax_traceback_filtering', 'off')

def scan_body(carry, x):
  checkify.check(jnp.all(x>0), "x needs to be strictly positive")
  jax.debug.breakpoint(backend="cli")
  return carry-1, x/carry

def f(xs):
  return lax.scan(scan_body, 5, xs)

f = checkify.checkify(f, errors=checkify.all_checks)
with jax.checking_leaks():
  jaxpr = jax.make_jaxpr(f)(jnp.arange(4) + 1)
print(jaxpr.literals)

# no error or breakpoint
err, out = f(jnp.arange(4)+1)
print("input jnp.arange(4)+1,    err:", err.get())

# error: let's enter postmortem debugging
err, out = f(jnp.arange(6))
print("input is jnp.arange(6),   err:", err.get())
