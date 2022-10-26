import traceback

import jax
from jax import core
from jax import random
from jax import lax
import jax.numpy as jnp

jax.config.update("jax_enable_custom_prng", True)

def f(key):
  k1, k2 = random.split(key)
  return random.normal(k1) + random.normal(k2)

jaxpr = jax.make_jaxpr(f)(random.PRNGKey(0)).jaxpr
core.check_jaxpr(jaxpr)

def f(key):
  def body(key, x):
    key, subkey = random.split(key)
    x = x + random.normal(subkey)
    return key, x
  return lax.scan(body, key, jnp.arange(5.))

jaxpr = jax.make_jaxpr(f)(random.PRNGKey(0)).jaxpr
core.check_jaxpr(jaxpr)

@jax.jit
def f(key):
  def body(key, x):
    x = x + random.normal(key) 
    # Key re-used on next iteration
    return key, x
  return lax.scan(body, key, jnp.arange(5.))

try:
  jaxpr = jax.make_jaxpr(f)(random.PRNGKey(0)).jaxpr
  core.check_jaxpr(jaxpr)
except core.JaxprTypeError:
  pass
  # traceback.print_exc()
else:
  print("UH OH!")

def f(key, x):
  def body(keys, x):
    k1, k2, k3 = keys
    x = x + random.normal(k1)
    return [k3, k1, k2], x
  k1, k2, k3 = random.split(key, 3)
  return lax.scan(body, [k1, k2, k3], x)

jaxpr = jax.make_jaxpr(f)(random.PRNGKey(0), jnp.arange(3.)).jaxpr
core.check_jaxpr(jaxpr)

try:
  jaxpr = jax.make_jaxpr(f)(random.PRNGKey(0), jnp.arange(4.)).jaxpr
  core.check_jaxpr(jaxpr)
except:
  pass
else:
  print("UH OH!")
