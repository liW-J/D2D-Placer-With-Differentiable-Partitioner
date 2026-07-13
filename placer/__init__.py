import sys
from pathlib import Path


def _add_dreamplace_legacy_import_path():
    # DREAMPlace modules still use absolute imports such as `import Params`.
    # Keep that compatibility in D2D instead of patching the submodule package.
    project_root = Path(__file__).resolve().parents[1]
    candidates = (
        project_root / "dreamplace",
        project_root / "thirdparty" / "DREAMPlace" / "dreamplace",
    )
    for package_dir in candidates:
        if (package_dir / "Params.py").is_file():
            for path in (package_dir.parent, package_dir):
                path_text = str(path)
                if path_text not in sys.path:
                    sys.path.insert(0, path_text)
            return


_add_dreamplace_legacy_import_path()
