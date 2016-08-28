#include <string>
#include "neo_ica/ica.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::tuple<py::array, py::array> ica(py::array& data, py::array& weights, py::array& sphere,
         int iter, unsigned int verbose, int nthreads, float rho, int fbatch, float theta, bool extended)
{
    //options
    neo_ica::options opt(iter, verbose, theta, rho, fbatch, nthreads, extended);
    //buffer
    py::buffer_info const & X = data.request();
    py::buffer_info const & W = weights.request();
    py::buffer_info const & Sphere = sphere.request();
    size_t NC = X.shape[0];
    size_t NF = X.shape[1];
    //dtype
    if(X.format == "f"){
        typedef float T;
        neo_ica::ica((T*)X.ptr, (T*)W.ptr, (T*)Sphere.ptr, NC, NF, opt);
    }
    else if(X.format == "d"){
        typedef double T;
        neo_ica::ica((T*)X.ptr, (T*)W.ptr, (T*)Sphere.ptr, NC, NF, opt);
    }
    return std::make_tuple(weights, sphere);
}

PYBIND11_PLUGIN(_ica) {
    py::module m("_ica", "C++ wrapper for neo-ica");
    m.def("ica", &ica,
          "Performs independent component analysis on the data provided",
          py::arg("data"), py::arg("weights"), py::arg("sphere"),
          py::arg("iter"), py::arg("verbose"),
          py::arg("nthreads"), py::arg("rho"),
          py::arg("fbatch"), py::arg("theta"), py::arg("extended"));

    py::module df = m.def_submodule("default", "Default values for parameters");
    using namespace neo_ica::dflt;
    df.attr("iter") = py::int_(iter);
    df.attr("verbose") = py::int_(verbose);
    df.attr("nthreads") = py::int_(nthreads);
    df.attr("rho") = py::float_(rho);
    df.attr("fbatch") = py::int_(fbatch);
    df.attr("theta") = py::float_(theta);
    df.attr("extended") = py::bool_(extended);
    return m.ptr();
}
