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


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
