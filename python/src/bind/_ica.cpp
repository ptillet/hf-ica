#include <string>
#include "neo_ica/ica.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::tuple<py::array, py::array, py::array> ica(py::array & data, py::array & weights, py::array & sphere,
         int iter, unsigned int verbosity, int nthreads, float rho, int fbatch, float theta)
{
    //options
    neo_ica::options opt;
    opt.iter = iter;
    opt.verbosity = verbosity;
    opt.nthreads = nthreads;
    opt.rho = rho;
    opt.fbatch = fbatch;
    opt.theta = theta;
    //buffer
    py::buffer_info X = data.request();
    py::buffer_info W = weights.request();
    py::buffer_info Sphere = sphere.request();
    //dtype
    if(X.format == "Zf"){
        typedef float T;
        neo_ica::ica<T>((T*)X.ptr, (T*)W.ptr, (T*)Sphere.ptr, X.shape[0], X.shape[1], opt);
    }
    else if(X.format == "Zd"){
        typedef double T;
        neo_ica::ica<T>((T*)X.ptr, (T*)W.ptr, (T*)Sphere.ptr, X.shape[0], X.shape[1], opt);
    }
    return std::make_tuple(data, weights, sphere);
}

PYBIND11_PLUGIN(_ica) {
    py::module m("_ica", "C++ wrapper for neo-ica");
    using namespace neo_ica::dflt;
    m.def("ica", &ica,
          "Performs independent component analysis on the data provided",
          py::arg("data"), py::arg("weights"), py::arg("sphere"),
          py::arg("iter")=iter, py::arg("verbosity")=verbosity,
          py::arg("theta")=theta, py::arg("rho")=rho, py::arg("fbatch")=fbatch,
          py::arg("nthreads")=nthreads);

    return m.ptr();
}
