#include "ctranslate2/models/model_factory.h"

#include <mutex>

#include "ctranslate2/models/whisper.h"

namespace ctranslate2 {
  namespace models {

    static void register_supported_models() {
      register_model<WhisperModel>("WhisperSpec");
    }

    std::shared_ptr<Model> create_model(const std::string& name) {
      static std::once_flag init_flag;
      std::call_once(init_flag, register_supported_models);

      return ModelFactory::get_instance().create_model(name);
    }

  }
}
