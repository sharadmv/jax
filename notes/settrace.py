import contextlib
import jax
import jax.numpy as jnp
import sys
from jax._src import traceback_util
  
def make_checkify_tracer(base_frame):
  def tracer(frame, event, arg=None):
    if event != "line":
      return tracer
    if frame != base_frame:
      return None
    jax.debug.breakpoint(backend="cli", ignore_frames=1)
    return tracer
  return tracer

@contextlib.contextmanager
def break_if_error():
  sys.settrace(lambda *args: None)
  frame = sys._getframe(2)
  tracer = make_checkify_tracer(frame)
  frame.f_trace = tracer
  yield
  sys.settrace(None)

  
def g(x):
  with break_if_error():
    y = jnp.sin(x)
    z = jnp.exp(y)
    z = z + 1
    return z
  
@jax.jit
def f(x):
  return g(g(x))
f(jnp.arange(5.))
