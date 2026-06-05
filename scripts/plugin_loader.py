import sys
import importlib
from pathlib import Path

"""
Scans the scripts directory for concrete implementations of PyProcessor.

This module dynamically searches for any Python file ending in 'processor.py' 
within its own directory and imports it. Doing so triggers the abstract base 
class's `__init_subclass__` hook, effectively auto-registering the plugins.
"""

# get the directory where this loader script lives
scripts_dir = Path(__file__).parent

# find all files ending in processor.py
for script_path in scripts_dir.glob("*processor.py"):
    # do not import this loader (if it matches somehow)
    if script_path.name == "plugin_loader.py":
        continue

    # retrieve the filename without the filename extension
    module_name = script_path.stem

    try:
        # dynamically import the module
        importlib.import_module(f".{module_name}", package=__package__)
        print(f"Successfully loaded plugin: {module_name}")
    except Exception as e:
        print(f"Failed to load plugin {module_name}: {e}", file=sys.stderr)
