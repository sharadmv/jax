# # Modules in JAX

import abc
import dataclasses

from jax._src.lax.control_flow import for_loop

import jax
import jax.numpy as jnp

Ref = for_loop.Ref
ref_get = for_loop.ref_get
ref_set = for_loop.ref_set

# # Testing `Ref`s

ref = Ref(jnp.array(0))
def f(ref, y):
  ref_set(ref, (), y)
  return
print("Old value:", ref.value)
jax.jit(f)(ref, 2)
print("New value:", ref.value)


# # Module

class ModuleMeta(abc.ABCMeta):
   def __new__(mcs, name, bases, dict_):
      cls = super().__new__(mcs, name, bases, dict_)
      cls = dataclasses.dataclass(eq=False, repr=False, frozen=False, init=True)(cls)
      jax.tree_util.register_pytree_node_class(cls)
      return cls

class Module(metaclass=ModuleMeta):

  def __setattr__(self, name, value):
    if hasattr(self, name):
      ref = getattr(self, name)
      if isinstance(ref, Ref):
        ref_set(ref, (), value)
        return
    super().__setattr__(name, value)

  def tree_flatten(self):
      dynamic_field_names = []
      dynamic_field_values = []
      static_field_names = []
      static_field_values = []
      for field_ in fields(self):
          name = field_.name
          try:
              value = self.__dict__[name]
          except KeyError:
              continue
          if field_.metadata.get("static", False):
              static_field_names.append(name)
              static_field_values.append(value)
          else:
              dynamic_field_names.append(name)
              dynamic_field_values.append(value)
      return tuple(dynamic_field_values), (
          tuple(dynamic_field_names),
          tuple(static_field_names),
          tuple(static_field_values),
      )

  @classmethod
  def tree_unflatten(cls, aux, dynamic_field_values):
      self = cls.__new__(cls)
      dynamic_field_names, static_field_names, static_field_values = aux
      for name, value in zip(dynamic_field_names, dynamic_field_values):
          object.__setattr__(self, name, value)
      for name, value in zip(static_field_names, static_field_values):
          object.__setattr__(self, name, value)
      return self

class Counter(Module):
  count: Ref

  def __init__(self):
    self.count = Ref(0)

  def __call__(self, x):
    self.count += 1
    return self.count[()] + x

counter = Counter()
def inc(x):
  return counter(x)
print(inc(1))
print(jax.jit(inc)(1))
