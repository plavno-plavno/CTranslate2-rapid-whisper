#include <pybind11/pybind11.h>

#include <ctranslate2/devices.h>
#include <ctranslate2/models/model.h>
#include <ctranslate2/random.h>
#include <ctranslate2/types.h>
#include <ctranslate2/utils.h>

#include "module.h"
#include "utils.h"


static std::unordered_set<std::string>
get_supported_compute_types(const std::string& device_str, const int device_index) {
  const auto device = ctranslate2::str_to_device(device_str);

  const bool support_bfloat16 = ctranslate2::mayiuse_bfloat16(device, device_index);
  const bool support_float16 = ctranslate2::mayiuse_float16(device, device_index);

  std::unordered_set<std::string> compute_types;
  compute_types.emplace("float32");

  if (support_bfloat16)
    compute_types.emplace("bfloat16");

  if (support_float16)
    compute_types.emplace("float16");

  return compute_types;
}


PYBIND11_MODULE(_ext, m)
{
  py::options options;
  options.disable_enum_members_docstring();

  m.def("contains_model", &ctranslate2::models::contains_model, py::arg("path"),
        "Helper function to check if a directory seems to contain a CTranslate2 model.");

  m.def("get_cuda_device_count", &ctranslate2::get_gpu_count,
        "Returns the number of visible GPU devices.");

  m.def("get_supported_compute_types", &get_supported_compute_types,
        py::arg("device"),
        py::arg("device_index")=0,
         R"pbdoc(
             Returns the set of supported compute types on a device.

             Arguments:
               device: Device name (cpu or cuda).
               device_index: Device index.

             Example:
                 >>> ctranslate2.get_supported_compute_types("cpu")
                 {'int16', 'float32', 'int8', 'int8_float32'}
                 >>> ctranslate2.get_supported_compute_types("cuda")
                 {'float32', 'int8_float16', 'float16', 'int8', 'int8_float32'}
         )pbdoc");

  m.def("set_random_seed", &ctranslate2::set_random_seed, py::arg("seed"),
        "Sets the seed of random generators.");

  ctranslate2::python::register_logging(m);
  ctranslate2::python::register_storage_view(m);
  ctranslate2::python::register_whisper(m);
}
