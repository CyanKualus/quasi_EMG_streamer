"""EMGcasting: hand-EMG analysis for Quasi-fMRI recordings.

The EMG half of NeuroCasting. :func:`analyze_batch` classifies every recording
and keeps the result in memory for the application to draw; :func:`process_batch`
also writes the figures and tables.
"""

from .core import (BatchResult, ProcessingConfig, analyze_batch, process_batch,
                   save_batch_outputs)

__all__ = ["ProcessingConfig", "BatchResult", "analyze_batch", "process_batch",
           "save_batch_outputs"]
