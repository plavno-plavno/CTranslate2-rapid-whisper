#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ctranslate2 {
  namespace python {

    void register_logging(py::module& m);
    void register_storage_view(py::module& m);
    void register_whisper(py::module& m);

  }
}
