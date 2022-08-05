import os
import functools
import re

import jax
import jax.numpy as jnp
from jax.config import config


config.update("jax_traceback_filtering", "off")
config.update("jax_platform_name", "cpu")

def set_host_device_count(n):
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+", "", xla_flags
    ).split()
    os.environ["XLA_FLAGS"] = " ".join(
        ["--xla_force_host_platform_device_count={}".format(n)] + xla_flags
    )

set_host_device_count(8)

@functools.partial(jax.pmap, axis_name="foo")
def f(x):
  def cond(i):
    return i < 10
  def body(i):
    return i + 1
  return jax.lax.while_loop(cond, body, x)

@functools.partial(jax.pmap, axis_name="foo")
def f(x):
  def cond(i):
    return i < 10
  def body(i):
    return i + 1
  return jax.lax.while_loop(cond, body, x)

print(f(jnp.arange(4.)))

with jax.disable_jit():
  print(f(jnp.arange(4.)))

# TODO:
# * process_call/map
# * nesting
