#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

int haroon_print() {
    std::cout << "Hello ffrom c++" << std::endl;
    return 0;
}

PYBIND11_MODULE(cuda_kernels, m) {
    m.def("haroon_print", &haroon_print, "hello worlld cpp");
}