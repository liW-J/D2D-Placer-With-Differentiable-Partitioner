#include <yaml-cpp/yaml.h>   
#include <string>
#include <unordered_map>

class PathConfig {
public:
    static PathConfig& getInstance() {
        static PathConfig instance;
        return instance;
    }

    void loadConfig(const std::string& configPath = "config/config.yaml") {
        try {
            YAML::Node config = YAML::LoadFile(configPath);
            
            auto sysInfo = config["SYSTEM_INFO"];
            osName = sysInfo["os_name"].as<std::string>();
            
            auto projectPaths = config["PROJECT_PATHS"];
            paths["PROJECT_ROOT"] = projectPaths["project_root"].as<std::string>();
            paths["SOURCE_DIR"] = projectPaths["source_dir"].as<std::string>();
            
            auto macros = config["MACROS"];
            maxThreads = macros["MAX_THREADS"].as<int>();
            bufferSize = macros["DEFAULT_BUFFER_SIZE"].as<int>();
        } catch (const YAML::Exception& e) {
            std::cerr << "Error loading config: " << e.what() << std::endl;
        }
    }

    std::string getPath(const std::string& key) const {
        auto it = paths.find(key);
        return it != paths.end() ? it->second : "";
    }

private:
    PathConfig() { loadConfig(); }
    
    std::string osName;
    std::unordered_map<std::string, std::string> paths;
    int maxThreads;
    int bufferSize;
}; 