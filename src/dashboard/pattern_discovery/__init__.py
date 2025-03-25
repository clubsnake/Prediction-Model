"""
Pattern discovery module for the dashboard.
Exposes functions for identifying, visualizing, and finding similar patterns in market data.
"""

import pandas as pd
import numpy as np

from .pattern_discovery_tab import PatternDiscoveryTab
from .pattern_management import PatternManager, discover_patterns_from_df

# Create global instances to maintain state
_pattern_manager = None
_pattern_tab = None

def get_pattern_manager():
    """Get or create the pattern manager instance"""
    global _pattern_manager
    if _pattern_manager is None:
        _pattern_manager = PatternManager()
    return _pattern_manager

def identify_patterns(df, window_size=20):
    """
    Identify patterns in the given market data
    
    Args:
        df: DataFrame with market data
        window_size: Size of window for pattern detection
        
    Returns:
        list: List of identified patterns
    """
    pattern_manager = get_pattern_manager()
    return pattern_manager.discover_patterns(
        df, 
        min_occurrences=max(3, window_size//5),
        significance_threshold=1.5
    )

def visualize_patterns(patterns, df):
    """
    Visualize the identified patterns
    
    Args:
        patterns: List of patterns to visualize
        df: DataFrame with market data
    """
    global _pattern_tab
    if _pattern_tab is None:
        _pattern_tab = PatternDiscoveryTab(df)
    
    if not patterns:
        import streamlit as st
        st.info("No patterns identified in the current data")
        return
        
    # Visualize each pattern
    for i, pattern in enumerate(patterns[:5]):  # Limit to 5 patterns
        _pattern_tab._render_pattern_card(pattern, is_active=True)
        
        # Add separator between patterns except after the last one
        if i < len(patterns[:5]) - 1:
            import streamlit as st
            st.markdown("---")

def find_similar_patterns(df, current_pattern, max_patterns=3):
    """
    Find patterns similar to the current pattern
    
    Args:
        df: DataFrame with market data
        current_pattern: Pattern or window of data to match
        max_patterns: Maximum number of patterns to return
        
    Returns:
        dict: Dictionary with similar patterns information
    """
    pattern_manager = get_pattern_manager()
    
    # If current_pattern is a numpy array, convert to pattern dict
    if isinstance(current_pattern, (np.ndarray, list)):
        # Create a temporary pattern dict from the array
        temp_pattern = {
            "id": "temp_pattern",
            "values": current_pattern,
            "name": "Current Pattern"
        }
        
        # Create sample patterns from the data
        all_patterns = []
        for i in range(0, len(df) - len(current_pattern), len(current_pattern) // 2):
            if i + len(current_pattern) < len(df) and "Close" in df.columns:
                values = df["Close"].iloc[i:i+len(current_pattern)].values
                all_patterns.append({
                    "id": f"sample_{i}",
                    "values": values,
                    "name": f"Pattern at index {i}",
                    "start_idx": i,
                    "end_idx": i + len(current_pattern)
                })
        
        # Calculate similarity between current pattern and all patterns
        similar_patterns = []
        for pattern in all_patterns:
            # Compare normalized patterns
            similarity = calculate_pattern_similarity(
                current_pattern / current_pattern[0] if current_pattern[0] != 0 else current_pattern,
                pattern["values"] / pattern["values"][0] if pattern["values"][0] != 0 else pattern["values"]
            )
            
            if similarity > 0.6:  # Minimum similarity threshold
                similar_patterns.append({
                    "id": pattern["id"],
                    "name": pattern["name"],
                    "similarity": similarity,
                    "values": pattern["values"],
                    "start_idx": pattern["start_idx"],
                    "end_idx": pattern["end_idx"]
                })
                
        # Sort by similarity and limit to max_patterns
        similar_patterns.sort(key=lambda x: x["similarity"], reverse=True)
        similar_patterns = similar_patterns[:max_patterns]
        
        return {
            "current_pattern": current_pattern,
            "patterns": similar_patterns
        }
    
    # If current_pattern is already a pattern dict, use pattern_manager
    else:
        similar_patterns = pattern_manager.find_similar_patterns(
            current_pattern, similarity_threshold=0.6)[:max_patterns]
        return {
            "current_pattern": current_pattern,
            "patterns": similar_patterns
        }

def calculate_pattern_similarity(pattern1, pattern2):
    """
    Calculate similarity between two patterns using dynamic time warping
    
    Args:
        pattern1: First pattern (numpy array or list)
        pattern2: Second pattern (numpy array or list)
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Simple Euclidean distance for now, could use DTW later
    if len(pattern1) != len(pattern2):
        # Resample to same length
        min_len = min(len(pattern1), len(pattern2))
        if isinstance(pattern1, np.ndarray):
            pattern1 = pattern1[:min_len]
        else:
            pattern1 = pattern1[:min_len]
            
        if isinstance(pattern2, np.ndarray):
            pattern2 = pattern2[:min_len]
        else:
            pattern2 = pattern2[:min_len]
    
    # Convert to numpy arrays
    p1 = np.array(pattern1)
    p2 = np.array(pattern2)
    
    # Calculate normalized Euclidean distance
    dist = np.sqrt(np.mean((p1 - p2) ** 2))
    
    # Convert to similarity score (1 = identical, 0 = completely different)
    similarity = 1 / (1 + dist)
    
    return similarity
