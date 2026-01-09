from pathlib import Path
import sys
import os


def get_project_root():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    
    current_file = Path(__file__).resolve()
    package_root = current_file.parent.parent
    
    project_root = package_root.parent
    
    if (project_root / "assets").exists() or (project_root / "models").exists():
        return project_root
    
    cwd = Path.cwd()
    if (cwd / "assets").exists() or (cwd / "models").exists():
        return cwd
    
    return project_root


def get_assets_path(filename):
    root = get_project_root()
    
    paths_to_check = [
        root / "assets" / filename,
        Path.cwd() / "assets" / filename,
        Path(__file__).parent.parent.parent / "assets" / filename,
    ]
    
    for path in paths_to_check:
        if path.exists():
            return str(path.resolve())
    
    return str(root / "assets" / filename)


def get_models_path(filename):
    root = get_project_root()
    
    paths_to_check = [
        root / "models" / filename,
        Path.cwd() / "models" / filename,
        Path(__file__).parent.parent.parent / "models" / filename,
    ]
    
    for path in paths_to_check:
        if path.exists():
            return str(path.resolve())
    
    return str(root / "models" / filename)
