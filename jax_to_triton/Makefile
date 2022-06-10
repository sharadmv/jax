default: custom_call.so
custom_call.so: custom_call.cc
	  g++ custom_call.cc -o custom_call.so -lcuda -shared -fPIC $(shell python3 -m pybind11 --includes)
