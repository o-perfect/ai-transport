"""Strait risk model: PortWatch transits -> weekly anomaly features -> 3-state HMM.

Plan: docs/strait-risk-plan.md. Validation report: `python -m strait_risk`.

    from strait_risk import assess
    assess("hormuz")["state"]   # 'Normal' | 'Reduced' | 'Closed'
"""

from .api import assess, assess_all, assess_many, get_model, weekly_table
from .data import CHOKEPOINTS, resolve
from .hmm import STATES

__all__ = ["assess", "assess_all", "assess_many", "get_model", "weekly_table",
           "CHOKEPOINTS", "resolve", "STATES"]
