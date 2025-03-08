"""
Pattern discovery module for cryptocurrency prediction dashboard.
This module contains functionality to discover, analyze, and track patterns in market data.
"""

from .pattern_discovery_tab import PatternDiscoveryTab, add_pattern_discovery_tab
from .pattern_management import PatternManager, discover_patterns_from_df

__all__ = ['PatternDiscoveryTab', 'add_pattern_discovery_tab', 'PatternManager', 'discover_patterns_from_df']
