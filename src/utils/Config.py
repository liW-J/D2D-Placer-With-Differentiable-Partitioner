import os
import yaml
from pathlib import Path
import platform
import sys
import tempfile

def get_system_info():
    return {
        "os_name": platform.system().lower(),
        "os_release": platform.release(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "architecture": platform.machine()
    }

def generate_paths_config():
    # 获取项目根目录（假设这个脚本在 scripts/ 目录下）
    project_root = Path(__file__).parent.parent.absolute()
    
    # 系统相关路径
    system_paths = {
        "temp_dir": tempfile.gettempdir(),
        "home_dir": str(Path.home()),
        "current_dir": str(Path.cwd())
    }
    
    # 项目相关路径
    project_paths = {
        "project_root": str(project_root),
        "source_dir": str(project_root / "src"),
        "build_dir": str(project_root / "build"),
        "config_dir": str(project_root / "config"),
        "test_dir": str(project_root / "test"),
        "thirdparty_dir": str(project_root / "thirdparty"),
    }
    
    # 运行时路径
    runtime_paths = {
        "result_dir": str(project_root / "result"),
    }
    
    # 系统信息
    sys_info = get_system_info()
    
    # 组合配置
    config = {
        "SYSTEM_INFO": sys_info,
        "SYSTEM_PATHS": system_paths,
        "PROJECT_PATHS": project_paths,
        "RUNTIME_PATHS": runtime_paths,
        
        # 其他宏定义
        "MACROS": {
            "MAX_THREADS": os.cpu_count(),
            "DEFAULT_BUFFER_SIZE": 8192,
            "DEBUG_LEVEL": 1,
        }
    }
    
    # 根据操作系统添加特定配置
    if sys_info["os_name"] == "linux":
        config["LINUX_SPECIFIC"] = {
            "lib_path": "/usr/lib",
            "include_path": "/usr/include",
            "bin_path": "/usr/bin"
        }
    
    return config

def save_config(config):
    # 生成 YAML 配置
    yaml_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

if __name__ == "__main__":
    config = generate_paths_config()
    save_config(config)
    print("config successed! save to: config/config.yaml") 