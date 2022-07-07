default: jax_triton/custom_call.so

jax_triton/custom_call.so: custom_call.cc
	  g++ --std=c++17 custom_call.cc -o jax_triton/custom_call.so -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -shared -fPIC $(shell python3 -m pybind11 --includes) -I/usr/local/cuda/include -lcuda -fpermissive

clean:
	rm custom_call.so
