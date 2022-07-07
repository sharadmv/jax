default: custom_call.so

custom_call.so: custom_call.cc
	  g++ custom_call.cc -o custom_call.so -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -shared -fPIC $(shell python3 -m pybind11 --includes) -I/usr/local/cuda/include -lcuda -fpermissive

clean:
	rm custom_call.so
