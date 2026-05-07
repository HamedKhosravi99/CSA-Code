"""
CSA Multi-Domain Experiment Framework.

Provides shared abstractions for running CSA deployment-time safety
experiments across medical, financial, legal, and autonomous agent domains.
"""

from domains.base import DomainStream, DomainVerifier
from domains.surrogate import OnlineSurrogate
from domains.baselines import AlwaysAct, FixedThreshold, NaiveTuning
