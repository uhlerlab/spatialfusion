from importlib.resources import files, as_file
from pathlib import Path

def resolve_pkg_ckpt(relpath: str) -> Path:
    """
    Resolve a file committed under spatialfusion/src/spatialfusion/data/<relpath>
    to a real filesystem path, regardless of whether the package is in a wheel/zip.
    """
    resource = files("spatialfusion").joinpath(f"data/{relpath}")
    with as_file(resource) as p:
        return Path(p)
