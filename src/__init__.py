"""
src package initialization
"""

from .data_processing import DataProcessor
from .utils import (
    create_directory,
    save_model,
    load_model,
    save_results,
    load_results,
    set_plot_style
)

__version__ = '1.0.0'
__all__ = [
    'DataProcessor',
    'create_directory',
    'save_model',
    'load_model',
    'save_results',
    'load_results',
    'set_plot_style'
]
