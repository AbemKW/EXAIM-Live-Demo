"""Compatibility shim: exaid.exaid re-exports EXAIM as EXAID."""

from exaim_core.exaim import EXAIM

# Alias for backward compatibility
EXAID = EXAIM

__all__ = ['EXAID', 'EXAIM']
