"""The Zet interface: `zet ui`, `python -m zet.ui`, or double-click Zet.bat."""
from .server import App, main, serve

__all__ = ["App", "main", "serve"]
