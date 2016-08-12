#include <string>
#include "neo_ica/ica.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::tuple<py::array, py::array, py::array> ica(py::array & data,
         int iter, unsigned int verbosity, int nthreads, float rho, int fbatch, float theta)
{
    //options
    neo_ica::options opt(iter, verbosity, theta, rho, fbatch, nthreads);
    //buffer
    py::buffer_info X = data.request();
    size_t NC = X.shape[0];
    size_t NF = X.shape[1];

    py::buffer_info W(malloc(NC*NC), X.itemsize, X.format, 2, {NC, NC}, {X.itemsize, NC*X.itemsize});
    py::buffer_info Sphere(malloc(NC*NC), W.itemsize, W.format, W.ndim, W.shape, W.strides);
    //dtype
    if(X.format == "Zf"){
        typedef float T;
        neo_ica::ica((T*)X.ptr, (T*)W.ptr, (T*)Sphere.ptr, NC, NF, opt);
    }
    else if(X.format == "Zd"){
        typedef double T;
        neo_ica::ica((T*)X.ptr, (T*)W.ptr, (T*)Sphere.ptr, NC, NF, opt);
    }
    return std::make_tuple(data, py::array(W), py::array(Sphere));
}

PYBIND11_PLUGIN(_ica) {
    py::module m("_ica", "C++ wrapper for neo-ica");
    using namespace neo_ica::dflt;
    m.def("ica", &ica,
          "Performs independent component analysis on the data provided",
          py::arg("data"),
          py::arg("iter")=iter, py::arg("verbosity")=verbosity,
          py::arg("theta")=theta, py::arg("rho")=rho, py::arg("fbatch")=fbatch,
          py::arg("nthreads")=nthreads);

    return m.ptr();
}
