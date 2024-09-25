//
// Created by systembug on 24. 9. 24.
//

#ifndef IDEA_DNN_INFER_CONFIGMANAGER
#define IDEA_DNN_INFER_CONFIGMANAGER

#include <string>

namespace idea::dnn::infer {
    class InferConfig;
    class ConfigManager {
    public:
        ConfigManager() = default;
        ~ConfigManager() = default;

    public:
        InferConfig* parse(const std::string& path);
    };
}


#endif //DNN_CONFIGMANAGER_H
