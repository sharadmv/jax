# Copyright 2022 Google LLC
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
from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax.config import config
from jax._src import api
from jax._src import state
from jax._src import test_util as jtu
import numpy as np

config.parse_flags_with_absl()

prev_xla_flags = None


def setUpModule():
  global prev_xla_flags
  # This will control the CPU devices. On TPU we always have 2 devices
  prev_xla_flags = jtu.set_host_platform_device_count(2)


# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  prev_xla_flags()


class RefCompatibilityTest(jtu.JaxTestCase):

  def test_can_make_ref(self):
    api.make_ref(jnp.ones(5))

  def test_ref_has_shape_and_dtype(self):
    x = jnp.ones(5)
    ref = api.make_ref(x)
    self.assertTupleEqual(ref.shape, (5,))
    self.assertEqual(ref.dtype, x.dtype)

  def test_can_pass_ref_into_jitted_functions(self):
    @jax.jit
    def f(x):
      return x
    ref = api.make_ref(jnp.ones(5))
    out = f(ref)
    self.assertIsInstance(out, state.Ref)
    self.assertTupleEqual(out.shape, (5,))
    self.assertEqual(out.dtype, ref.dtype)

  def test_can_stage_out_stateful_functions(self):
    def f(ref):
      x = state.ref_get(ref, ())
      state.ref_set(ref, (), 1.)
      return x
    ref = api.make_ref(jnp.array(0.))
    jaxpr = jax.make_jaxpr(f)(ref)
    self.assertSetEqual(jaxpr.effects, {state.State})

  def test_can_call_methods_on_refs(self):
    @jax.jit
    def f(ref):
      x = ref.get()
      ref.set(x * 2.)
      return x * 2.
    ref = api.make_ref(jnp.array(1.))
    self.assertEqual(f(ref), 2.)
    self.assertEqual(ref.get(), 2.)

  def test_can_get_multiple_references_to_same_value(self):
    @jax.jit
    def f(ref):
      x = ref.get()
      ref.set(x * 2.)
      return ref
    ref = api.make_ref(jnp.array(1.))
    ref2 = f(ref)
    self.assertEqual(ref2.get(), 2.)
    self.assertEqual(ref.get(), 2.)
    ref2.set(1.)
    self.assertEqual(ref2.get(), 1.)
    self.assertEqual(ref.get(), 1.)

  def test_state_with_vmap(self):
    ref = api.make_ref(jnp.ones(4))

    @jax.vmap
    def foo(x):
      ref.set(x)
    out = foo(jnp.arange(4.))
    print("OUT", ref.get())
    assert False

  # def test_can_create_ref_in_jitted_function(self):
  #   @jax.jit
  #   def f(x):
  #     ref = api.make_ref(x)
  #     return ref.get()
  #   print(jax.make_jaxpr(f)(jnp.int32(2)))
  #   ref = f(2.)
  #   self.assertEqual(ref.get(), 2.)
  #   ref.set(1.)
  #   self.assertEqual(ref.get(), 1.)

class RunStateTest(jtu.JaxTestCase):

  def test_run_state_simple(self):

    def f(ref):
      ref.set(2.)

    out = state.run_state(f)(1.)
    self.assertTupleEqual(out, (2.,))

  def test_run_state_simple_closed_over_ref(self):

    outer_ref = api.make_ref(3.)
    def f(ref):
      ref.set(2.)
      outer_ref.set(4.)

    out = state.run_state(f)(1.)
    self.assertTupleEqual(out, (2.,))
    self.assertEqual(outer_ref.get(), 4.)

  def test_run_state_simple_closed_over_ref_in_jit(self):

    def f(outer_ref):
      def g(ref):
        ref.set(2.)
        outer_ref.set(4.)
      state.run_state(g)(3.)
    out = jax.jit(state.run_state(f))(1.)
    self.assertTupleEqual(out, (4.,))

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
