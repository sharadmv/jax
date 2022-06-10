// g++ custom_call.cc -o custom_call.so -lcuda -shared -fPIC $(python -m pybind11 --includes)
#include <iostream>
#include <string>

#include <pybind11/pybind11.h>
#include "cuda.h"

namespace py = pybind11;


void do_custom_call(CUstream stream, void** buffers,
		char* opaque, size_t opaque_len) {
	uint64_t descriptor = *reinterpret_cast<uint64_t*>(opaque);
	CUfunction kernel = reinterpret_cast<CUfunction>(descriptor);
	int grid_0 = 1;
	int grid_1 = 1;
	int grid_2 = 1;
	std::string params;
	params.resize(8 * 3);
	char* params_ptr = &params[0];
	size_t params_size;
	params_ptr = (char*)(((uintptr_t)params_ptr + 7) & (-8));
	std::memcpy(params_ptr, &buffers[0], 8);
	params_ptr += 8;
	params_ptr = (char*)(((uintptr_t)params_ptr + 7) & (-8));
	std::memcpy(params_ptr, &buffers[1], 8);
	params_ptr += 8;
	params_ptr = (char*)(((uintptr_t)params_ptr + 7) & (-8));
	std::memcpy(params_ptr, &buffers[2], 8);
	params_ptr += 8;
	params_size = (std::ptrdiff_t)(params_ptr - &params[0]);
	void* config[] = {
	  CU_LAUNCH_PARAM_BUFFER_POINTER, params.data(),
	  CU_LAUNCH_PARAM_BUFFER_SIZE, &params_size,
	  CU_LAUNCH_PARAM_END
	};
	CUresult result = cuLaunchKernel(kernel, grid_0, grid_1, grid_2, 4 * 32, 1, 1, 0, stream, nullptr, config);
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(reinterpret_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

PYBIND11_MODULE(custom_call, m) {
	m.def("get_custom_call", [](){ return EncapsulateFunction(do_custom_call); });
}

