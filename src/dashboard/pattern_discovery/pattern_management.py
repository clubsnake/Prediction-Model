import datetime
import json
import os
import uuid

import numpy as np
import pandas as pd

from config.config_loader import DATA_DIR


class PatternManager:
    """
    Manages pattern storage, retrieval, similarity, and archiving.
    This class handles the backend pattern management for the Pattern Discovery Tab.
    """

    def __init__(self, patterns_dir=None, archive_dir=None):
        """
        Initialize the pattern manager

        Args:
            patterns_dir: Directory to store active patterns
            archive_dir: Directory to store archived patterns
        """
        # Update to store patterns in Models directory
        self.patterns_dir = patterns_dir or os.path.join(
            DATA_DIR, "Models", "discovered_patterns"
        )
        self.archive_dir = archive_dir or os.path.join(self.patterns_dir, "archive")

        # Create directories if they don't exist
        os.makedirs(self.patterns_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)

        # Create archive index if it doesn't exist
        archive_index_path = os.path.join(self.archive_dir, "archive_index.json")
        if not os.path.exists(archive_index_path):
            with open(archive_index_path, "w") as f:
                json.dump({"patterns": []}, f)

    def save_pattern(self, pattern, update_if_exists=True):
        """
        Save a pattern to the patterns directory

        Args:
            pattern: Pattern dictionary to save
            update_if_exists: If True, update existing pattern with same ID

        Returns:
            str: Pattern ID that was saved
        """
        # Make sure pattern has an ID
        if "id" not in pattern:
            pattern["id"] = self._generate_pattern_id(pattern)

        # Make sure pattern has a timestamp
        if "timestamp" not in pattern:
            pattern["timestamp"] = datetime.datetime.now().isoformat()

        # Check if pattern already exists
        existing_pattern = self.get_pattern(pattern["id"])
        if existing_pattern and update_if_exists:
            pattern = self._merge_patterns(existing_pattern, pattern)

        # Save the pattern
        pattern_path = os.path.join(self.patterns_dir, f"{pattern['id']}.json")
        with open(pattern_path, "w") as f:
            json.dump(pattern, f, indent=2)

        return pattern["id"]

    def get_pattern(self, pattern_id):
        """
        Get a pattern by ID

        Args:
            pattern_id: ID of the pattern to get

        Returns:
            dict: Pattern dictionary or None if not found
        """
        pattern_path = os.path.join(self.patterns_dir, f"{pattern_id}.json")
        if os.path.exists(pattern_path):
            with open(pattern_path, "r") as f:
                return json.load(f)

        # Check archive if not found in active patterns
        archived_pattern = self._get_archived_pattern(pattern_id)
        return archived_pattern

    def list_patterns(self, include_archived=False):
        """
        List all patterns

        Args:
            include_archived: If True, include archived patterns

        Returns:
            list: List of pattern dictionaries
        """
        patterns = []

        # Get active patterns
        for filename in os.listdir(self.patterns_dir):
            if filename.endswith(".json"):
                pattern_path = os.path.join(self.patterns_dir, filename)
                try:
                    with open(pattern_path, "r") as f:
                        pattern = json.load(f)
                        patterns.append(pattern)
                except (json.JSONDecodeError, IOError):
                    # Skip invalid files
                    continue

        # Get archived patterns if requested
        if include_archived:
            archive_index_path = os.path.join(self.archive_dir, "archive_index.json")
            if os.path.exists(archive_index_path):
                try:
                    with open(archive_index_path, "r") as f:
                        archive_index = json.load(f)
                        for archived_pattern_info in archive_index.get("patterns", []):
                            pattern_id = archived_pattern_info.get("id")
                            if pattern_id:
                                archived_pattern = self._get_archived_pattern(
                                    pattern_id
                                )
                                if archived_pattern:
                                    patterns.append(archived_pattern)
                except (json.JSONDecodeError, IOError):
                    # Skip if archive index is invalid
                    pass

        return patterns

    def archive_pattern(self, pattern_id, reason="manual"):
        """
        Archive a pattern

        Args:
            pattern_id: ID of the pattern to archive
            reason: Reason for archiving (manual, performance, age, etc.)

        Returns:
            bool: True if successful, False otherwise
        """
        # Get the pattern
        pattern = self.get_pattern(pattern_id)
        if not pattern:
            return False

        # Add archive metadata
        pattern["archived"] = True
        pattern["archive_timestamp"] = datetime.datetime.now().isoformat()
        pattern["archive_reason"] = reason

        # Save to archive directory
        archive_path = os.path.join(self.archive_dir, f"{pattern_id}.json")
        with open(archive_path, "w") as f:
            json.dump(pattern, f, indent=2)

        # Update archive index
        archive_index_path = os.path.join(self.archive_dir, "archive_index.json")
        try:
            with open(archive_index_path, "r") as f:
                archive_index = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            archive_index = {"patterns": []}

        # Add to index if not already there
        if pattern_id not in [p.get("id") for p in archive_index["patterns"]]:
            archive_index["patterns"].append(
                {
                    "id": pattern_id,
                    "name": pattern.get("name", "Unnamed Pattern"),
                    "timestamp": pattern.get("timestamp", ""),
                    "archive_timestamp": pattern["archive_timestamp"],
                    "reason": reason,
                }
            )

            with open(archive_index_path, "w") as f:
                json.dump(archive_index, f, indent=2)

        # Remove from active patterns
        active_path = os.path.join(self.patterns_dir, f"{pattern_id}.json")
        if os.path.exists(active_path):
            os.remove(active_path)

        return True

    def unarchive_pattern(self, pattern_id):
        """
        Unarchive a pattern

        Args:
            pattern_id: ID of the pattern to unarchive

        Returns:
            bool: True if successful, False otherwise
        """
        # Get the archived pattern
        pattern = self._get_archived_pattern(pattern_id)
        if not pattern:
            return False

        # Remove archive metadata
        if "archived" in pattern:
            del pattern["archived"]
        if "archive_timestamp" in pattern:
            del pattern["archive_timestamp"]
        if "archive_reason" in pattern:
            del pattern["archive_reason"]

        # Save to active directory
        active_path = os.path.join(self.patterns_dir, f"{pattern_id}.json")
        with open(active_path, "w") as f:
            json.dump(pattern, f, indent=2)

        # Update archive index
        archive_index_path = os.path.join(self.archive_dir, "archive_index.json")
        try:
            with open(archive_index_path, "r") as f:
                archive_index = json.load(f)

                # Remove from index
                archive_index["patterns"] = [
                    p for p in archive_index["patterns"] if p.get("id") != pattern_id
                ]

                with open(archive_index_path, "w") as f:
                    json.dump(archive_index, f, indent=2)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

        # Remove from archive
        archive_path = os.path.join(self.archive_dir, f"{pattern_id}.json")
        if os.path.exists(archive_path):
            os.remove(archive_path)

        return True

    def list_archived_patterns(self):
        """
        List all archived patterns

        Returns:
            list: List of archived pattern dictionaries
        """
        archived_patterns = []

        # Get archived patterns from index
        archive_index_path = os.path.join(self.archive_dir, "archive_index.json")
        if os.path.exists(archive_index_path):
            try:
                with open(archive_index_path, "r") as f:
                    archive_index = json.load(f)
                    for archived_pattern_info in archive_index.get("patterns", []):
                        pattern_id = archived_pattern_info.get("id")
                        if pattern_id:
                            archived_pattern = self._get_archived_pattern(pattern_id)
                            if archived_pattern:
                                archived_patterns.append(archived_pattern)
            except (json.JSONDecodeError, IOError):
                # Skip if archive index is invalid
                pass

        return archived_patterns

    def auto_archive_patterns(self, age_threshold_days=180, performance_threshold=0.3):
        """
        Auto-archive patterns based on age and performance

        Args:
            age_threshold_days: Archive patterns older than this many days
            performance_threshold: Archive patterns with reliability below this value

        Returns:
            list: List of pattern IDs that were archived
        """
        archived_patterns = []

        # Get all active patterns
        patterns = self.list_patterns(include_archived=False)

        # Get current time
        now = datetime.datetime.now()

        # Check each pattern
        for pattern in patterns:
            pattern_id = pattern.get("id")
            if not pattern_id:
                continue

            # Check age
            timestamp_str = pattern.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.datetime.fromisoformat(timestamp_str)
                    age_days = (now - timestamp).days
                    if age_days > age_threshold_days:
                        self.archive_pattern(pattern_id, reason=f"age: {age_days} days")
                        archived_patterns.append(pattern_id)
                        continue
                except ValueError:
                    # Invalid timestamp format
                    pass

            # Check performance
            reliability = pattern.get("reliability", 0)
            if reliability < performance_threshold:
                self.archive_pattern(
                    pattern_id, reason=f"low reliability: {reliability}"
                )
                archived_patterns.append(pattern_id)

        return archived_patterns

    def find_similar_patterns(self, pattern, similarity_threshold=0.75):
        """
        Find patterns similar to the given pattern

        Args:
            pattern: Pattern dictionary to compare against
            similarity_threshold: Threshold for similarity score

        Returns:
            list: List of similar pattern dictionaries
        """
        similar_patterns = []

        # Get all patterns
        patterns = self.list_patterns(include_archived=True)

        # Compare with each pattern
        for existing_pattern in patterns:
            # Skip same pattern
            if existing_pattern.get("id") == pattern.get("id"):
                continue

            # Calculate similarity
            similarity = self._calculate_pattern_similarity(pattern, existing_pattern)

            # Add to results if similar enough
            if similarity >= similarity_threshold:
                similar_patterns.append(
                    {"pattern": existing_pattern, "similarity": similarity}
                )

        # Sort by similarity
        similar_patterns.sort(key=lambda x: x["similarity"], reverse=True)

        return similar_patterns

    def detect_pattern_in_data(self, pattern, data_df):
        """
        Detect occurrences of a pattern in data

        Args:
            pattern: Pattern dictionary to detect
            data_df: DataFrame with market data

        Returns:
            list: List of indices where pattern was detected
        """
        occurrences = []

        # Get pattern conditions
        conditions = pattern.get("conditions", [])
        if not conditions:
            return occurrences

        # Check each row
        for i in range(len(data_df)):
            # Check all conditions
            conditions_met = True
            for condition in conditions:
                indicator = condition.get("indicator")
                operator = condition.get("operator")
                value = condition.get("value")

                # Skip if condition is incomplete
                if not (indicator and operator and value is not None):
                    continue

                # Resolve indicator column
                col = self._resolve_indicator_column(indicator, data_df)
                if col is None:
                    conditions_met = False
                    break

                # Get actual value
                try:
                    actual = data_df.iloc[i][col]
                except (IndexError, KeyError):
                    conditions_met = False
                    break

                # Compare based on operator
                if operator == ">":
                    if not (actual > value):
                        conditions_met = False
                        break
                elif operator == ">=":
                    if not (actual >= value):
                        conditions_met = False
                        break
                elif operator == "<":
                    if not (actual < value):
                        conditions_met = False
                        break
                elif operator == "<=":
                    if not (actual <= value):
                        conditions_met = False
                        break
                elif operator == "==":
                    if not (actual == value):
                        conditions_met = False
                        break
                elif operator == "!=":
                    if not (actual != value):
                        conditions_met = False
                        break

            # Add occurrence if all conditions met
            if conditions_met:
                occurrences.append(i)

        return occurrences

    def get_active_patterns(self, data_df):
        """
        Get patterns that are currently active in the data

        Args:
            data_df: DataFrame with market data

        Returns:
            list: List of active pattern dictionaries
        """
        active_patterns = []

        # Get all patterns
        patterns = self.list_patterns(include_archived=False)

        # Check each pattern
        for pattern in patterns:
            # Detect occurrences in the last rows
            last_n_rows = min(5, len(data_df))
            recent_data = data_df.iloc[-last_n_rows:]
            occurrences = self.detect_pattern_in_data(pattern, recent_data)

            # Add to active patterns if recently detected
            if occurrences:
                # Add detection info to pattern
                pattern["last_activation"] = datetime.datetime.now().isoformat()
                pattern["last_occurrence_idx"] = occurrences[-1]
                active_patterns.append(pattern)

        return active_patterns

    def discover_patterns(self, data_df, min_occurrences=5, significance_threshold=1.5):
        """
        Discover patterns in data

        Args:
            data_df: DataFrame with market data
            min_occurrences: Minimum number of occurrences for a pattern to be valid
            significance_threshold: Minimum z-score for a pattern to be significant

        Returns:
            list: List of discovered pattern dictionaries
        """
        # Use the helper function
        return discover_patterns_from_df(
            data_df,
            lookback=min(len(data_df), 90),
            min_occurrences=min_occurrences,
            min_zscore=significance_threshold,
        )

    def _get_archived_pattern(self, pattern_id):
        """
        Get an archived pattern by ID

        Args:
            pattern_id: ID of the pattern to get

        Returns:
            dict: Pattern dictionary or None if not found
        """
        archive_path = os.path.join(self.archive_dir, f"{pattern_id}.json")
        if os.path.exists(archive_path):
            try:
                with open(archive_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # Skip invalid files
                pass
        return None

    def _generate_pattern_id(self, pattern):
        """
        Generate a unique ID for a pattern

        Args:
            pattern: Pattern dictionary

        Returns:
            str: Unique ID
        """
        # Use pattern name if available
        name = pattern.get("name", "")

        # Generate UUID
        id_base = str(uuid.uuid4())[:8]

        # Combine name and UUID
        if name:
            # Convert name to slug
            slug = name.lower().replace(" ", "_")
            # Remove non-alphanumeric characters
            slug = "".join(c for c in slug if c.isalnum() or c == "_")
            # Limit length
            slug = slug[:20]
            return f"{slug}_{id_base}"
        else:
            return f"pattern_{id_base}"

    def _merge_patterns(self, existing_pattern, new_pattern):
        """
        Merge two pattern dictionaries

        Args:
            existing_pattern: Existing pattern dictionary
            new_pattern: New pattern dictionary

        Returns:
            dict: Merged pattern dictionary
        """
        # Start with existing pattern
        merged = existing_pattern.copy()

        # Override with non-null values from new pattern
        for key, value in new_pattern.items():
            if value is not None:
                merged[key] = value

        # Special handling for occurrences
        if "occurrences" in new_pattern and "occurrences" in existing_pattern:
            merged["occurrences"] = max(
                existing_pattern["occurrences"], new_pattern["occurrences"]
            )

        # Special handling for historical examples
        if (
            "historical_examples" in new_pattern
            and "historical_examples" in existing_pattern
        ):
            # Combine examples
            examples = (
                existing_pattern["historical_examples"]
                + new_pattern["historical_examples"]
            )
            # Remove duplicates
            unique_examples = []
            seen_dates = set()
            for example in examples:
                date = example.get("date")
                if date and date not in seen_dates:
                    seen_dates.add(date)
                    unique_examples.append(example)
            merged["historical_examples"] = unique_examples

        return merged

    def _calculate_pattern_similarity(self, pattern1, pattern2):
        """
        Calculate similarity between two patterns

        Args:
            pattern1: First pattern dictionary
            pattern2: Second pattern dictionary

        Returns:
            float: Similarity score between 0 and 1
        """
        # Extract pattern components for comparison
        components = []

        # Compare conditions
        conditions1 = pattern1.get("conditions", [])
        conditions2 = pattern2.get("conditions", [])

        if conditions1 and conditions2:
            # Count matching conditions
            matches = 0
            total = max(len(conditions1), len(conditions2))

            for c1 in conditions1:
                for c2 in conditions2:
                    # Check if indicators match
                    if c1.get("indicator") == c2.get("indicator"):
                        # Check if operators and values are similar
                        if c1.get("operator") == c2.get("operator"):
                            # Compare values
                            v1 = c1.get("value")
                            v2 = c2.get("value")
                            if v1 is not None and v2 is not None:
                                try:
                                    # Calculate similarity based on value difference
                                    v1 = float(v1)
                                    v2 = float(v2)
                                    value_similarity = 1 - min(
                                        abs(v1 - v2) / max(abs(v1), abs(v2), 1), 1
                                    )
                                    matches += value_similarity
                                except (ValueError, TypeError):
                                    # Non-numeric values, check equality
                                    if v1 == v2:
                                        matches += 1
                            elif v1 == v2:  # Both None or equal
                                matches += 1
                        else:
                            # Different operators, partial match
                            matches += 0.5

            condition_similarity = matches / total if total > 0 else 0
            components.append(condition_similarity)

        # Compare expected returns
        r1 = pattern1.get("expected_return")
        r2 = pattern2.get("expected_return")

        if r1 is not None and r2 is not None:
            try:
                r1 = float(r1)
                r2 = float(r2)
                # Calculate similarity based on relative difference
                return_similarity = 1 - min(abs(r1 - r2) / max(abs(r1), abs(r2), 1), 1)
                components.append(return_similarity)
            except (ValueError, TypeError):
                pass

        # Compare pattern descriptions
        d1 = pattern1.get("description", "")
        d2 = pattern2.get("description", "")

        if d1 and d2:
            text_similarity = self._text_similarity(d1, d2)
            components.append(text_similarity)

        # Calculate overall similarity
        if components:
            return sum(components) / len(components)
        else:
            return 0.0

    def _text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            float: Similarity score between 0 and 1
        """
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _resolve_indicator_column(self, indicator_name, df):
        """
        Resolve indicator name to DataFrame column

        Args:
            indicator_name: Name of the indicator
            df: DataFrame with market data

        Returns:
            str: Column name or None if not found
        """
        # Direct match
        if indicator_name in df.columns:
            return indicator_name

        # Case-insensitive match
        for col in df.columns:
            if col.lower() == indicator_name.lower():
                return col

        # Fuzzy match (contains)
        for col in df.columns:
            if indicator_name.lower() in col.lower():
                return col

        return None


# This class can be used by the PatternDiscoveryTab class
def discover_patterns_from_df(df, lookback=90, min_occurrences=5, min_zscore=1.5):
    """
    Discover patterns in a DataFrame using statistical analysis.
    This is a simplified version that looks for indicator threshold crossings.

    Args:
        df: DataFrame with market data and indicators
        lookback: Number of days to look back
        min_occurrences: Minimum number of occurrences for a pattern to be valid
        min_zscore: Minimum z-score for a pattern to be significant

    Returns:
        list: List of discovered pattern dictionaries
    """
    # Ensure we have price data
    if "Close" not in df.columns:
        return []

    # Limit to lookback period
    df = df.iloc[-lookback:].copy()

    # Calculate future returns (1-day, 3-day, 5-day)
    for days in [1, 3, 5]:
        df[f"return_{days}d"] = df["Close"].pct_change(days).shift(-days) * 100

    # Find potential indicators (numeric columns that aren't price/volume/returns)
    exclude_patterns = ["open", "high", "low", "close", "volume", "return", "date"]
    indicators = [
        col
        for col in df.columns
        if not any(pat in col.lower() for pat in exclude_patterns)
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Skip if no indicators
    if not indicators:
        return []

    # Discover patterns
    discovered_patterns = []

    # For each indicator, find significant thresholds
    for indicator in indicators:
        # Get indicator values
        values = df[indicator].dropna()

        # Skip if too few values
        if len(values) < min_occurrences * 2:
            continue

        # Try different percentiles as thresholds
        for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            threshold = np.percentile(values, percentile)

            # Check crossings above threshold
            above_mask = df[indicator] > threshold

            if above_mask.sum() >= min_occurrences:
                # Get average returns after crossing
                avg_return_1d = df.loc[above_mask, "return_1d"].mean()
                avg_return_3d = df.loc[above_mask, "return_3d"].mean()
                avg_return_5d = df.loc[above_mask, "return_5d"].mean()

                # Get standard deviation of regular returns
                std_return_1d = df["return_1d"].std()
                std_return_3d = df["return_3d"].std()
                std_return_5d = df["return_5d"].std()

                # Calculate z-scores
                zscore_1d = avg_return_1d / (std_return_1d + 1e-10)
                zscore_3d = avg_return_3d / (std_return_3d + 1e-10)
                zscore_5d = avg_return_5d / (std_return_5d + 1e-10)

                # Get the max z-score and corresponding period
                zscores = [zscore_1d, zscore_3d, zscore_5d]
                max_zscore_idx = np.argmax(np.abs(zscores))
                max_zscore = zscores[max_zscore_idx]
                periods = [1, 3, 5]
                best_period = periods[max_zscore_idx]

                # Only add significant patterns
                if abs(max_zscore) >= min_zscore:
                    pattern = {
                        "id": f"{indicator}_above_{percentile}pct",
                        "name": f"{indicator} Above {percentile}th Percentile",
                        "description": f"When {indicator} rises above {threshold:.2f}, price tends to move {'up' if max_zscore > 0 else 'down'} in the next {best_period} days.",
                        "conditions": [
                            {
                                "indicator": indicator,
                                "operator": ">",
                                "value": threshold,
                            }
                        ],
                        "expected_return": float(
                            df.loc[above_mask, f"return_{best_period}d"].mean()
                        ),
                        "reliability": float(
                            np.sign(df.loc[above_mask, f"return_{best_period}d"]).mean()
                            * 0.5
                            + 0.5
                        ),
                        "timeframe": f"{best_period}d",
                        "occurrences": int(above_mask.sum()),
                        "discovery_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "category": "Indicator Pattern",
                        "type": "indicator",
                        "statistics": {
                            "z_score": float(max_zscore),
                            "avg_return_1d": float(avg_return_1d),
                            "avg_return_3d": float(avg_return_3d),
                            "avg_return_5d": float(avg_return_5d),
                        },
                    }
                    discovered_patterns.append(pattern)

            # Check crossings below threshold
            below_mask = df[indicator] < threshold

            if below_mask.sum() >= min_occurrences:
                # Get average returns after crossing
                avg_return_1d = df.loc[below_mask, "return_1d"].mean()
                avg_return_3d = df.loc[below_mask, "return_3d"].mean()
                avg_return_5d = df.loc[below_mask, "return_5d"].mean()

                # Get standard deviation of regular returns
                std_return_1d = df["return_1d"].std()
                std_return_3d = df["return_3d"].std()
                std_return_5d = df["return_5d"].std()

                # Calculate z-scores
                zscore_1d = avg_return_1d / (std_return_1d + 1e-10)
                zscore_3d = avg_return_3d / (std_return_3d + 1e-10)
                zscore_5d = avg_return_5d / (std_return_5d + 1e-10)

                # Get the max z-score and corresponding period
                zscores = [zscore_1d, zscore_3d, zscore_5d]
                max_zscore_idx = np.argmax(np.abs(zscores))
                max_zscore = zscores[max_zscore_idx]
                periods = [1, 3, 5]
                best_period = periods[max_zscore_idx]

                # Only add significant patterns
                if abs(max_zscore) >= min_zscore:
                    pattern = {
                        "id": f"{indicator}_below_{percentile}pct",
                        "name": f"{indicator} Below {percentile}th Percentile",
                        "description": f"When {indicator} falls below {threshold:.2f}, price tends to move {'up' if max_zscore > 0 else 'down'} in the next {best_period} days.",
                        "conditions": [
                            {
                                "indicator": indicator,
                                "operator": "<",
                                "value": threshold,
                            }
                        ],
                        "expected_return": float(
                            df.loc[below_mask, f"return_{best_period}d"].mean()
                        ),
                        "reliability": float(
                            np.sign(df.loc[below_mask, f"return_{best_period}d"]).mean()
                            * 0.5
                            + 0.5
                        ),
                        "timeframe": f"{best_period}d",
                        "occurrences": int(below_mask.sum()),
                        "discovery_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "category": "Indicator Pattern",
                        "type": "indicator",
                        "statistics": {
                            "z_score": float(max_zscore),
                            "avg_return_1d": float(avg_return_1d),
                            "avg_return_3d": float(avg_return_3d),
                            "avg_return_5d": float(avg_return_5d),
                        },
                    }
                    discovered_patterns.append(pattern)

    return discovered_patterns
