"""
General Playground Data Format Utility Module

This module provides utilities for converting various JSON data formats to playground input format.
The playground format is used for model evaluation and testing. This is a flexible utility that
can work with different data structures and field mappings.

Author: Bin Zhu
Date: August 6, 2025
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FieldMapping:
    """Simple configuration for identifying the utterance field in input data."""

    utterance_field: str = "utterance"  # Default field name for utterance

    # Optional: alternative field names to try if primary utterance_field is not found
    utterance_alternatives: List[str] = field(
        default_factory=lambda: ["text", "content", "query", "input", "message"]
    )


@dataclass
class PlaygroundEntry:
    """Data class representing a single playground entry."""

    utterance: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionConfig:
    """Configuration for data conversion behavior."""

    default_source: str = "general-data"
    default_segment: str = "default"
    filter_function: Optional[Callable[[Dict[str, Any]], bool]] = None


class PlaygroundDataConverter:
    """
    General converter for transforming various JSON data formats to playground input format.

    The playground INPUT format structure:
    [
      {
        "input": {
          "parameters": "{\"utterance\": \"...\", \"field1\": \"...\", \"field2\": \"...\", ...}"
        }
      }
    ]

    Playground only requires "utterance" for processing. All other fields are preserved
    metadata from the original input so that the original JSON structure can be restored
    after playground processing.

    This converter can handle:
    - Flat JSON data (list of objects)
    - Grouped JSON data (dictionary with categories as keys)
    - Custom field mappings
    - Flexible filtering
    """

    def __init__(
        self,
        field_mapping: Optional[FieldMapping] = None,
        conversion_config: Optional[ConversionConfig] = None,
    ):
        """
        Initialize the converter with field mappings and configuration.

        Args:
            field_mapping: Configuration for mapping source fields to playground format
            conversion_config: Configuration for conversion behavior
        """
        self.field_mapping = field_mapping or FieldMapping()
        self.config = conversion_config or ConversionConfig()

    def convert_data_to_playground(
        self,
        data: Union[Dict[str, List[Dict]], List[Dict], Dict],
        filter_function: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert various data formats to playground format.

        Args:
            data: Input data - can be grouped dict, flat list, or single dict
            filter_function: Optional function to filter entries

        Returns:
            List of playground-formatted entries
        """
        # Normalize data to list of dictionaries
        normalized_data = self._normalize_input_data(data)

        # Apply filtering
        if filter_function:
            normalized_data = [
                item for item in normalized_data if filter_function(item)
            ]
        elif self.config.filter_function:
            normalized_data = [
                item for item in normalized_data if self.config.filter_function(item)
            ]

        playground_entries = []

        for item_data in normalized_data:
            playground_entry = self._create_playground_entry_from_data(item_data)
            if playground_entry:
                playground_entries.append(playground_entry)

        logger.info(f"Converted {len(playground_entries)} entries to playground format")
        return playground_entries

    def convert_classification_data_to_playground(
        self,
        classification_data: Dict[str, List[Dict]],
        filter_classification: Optional[str] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Convert classification data to playground format (backward compatibility).

        Args:
            classification_data: Dictionary with category names as keys and utterance lists as values
            filter_classification: If specified, only include utterances with this classification
            include_metadata: Whether to include classification metadata in results (ignored - always preserved)

        Returns:
            List of playground-formatted entries
        """

        # Create a filter function for classification
        def classification_filter(item: Dict[str, Any]) -> bool:
            if not filter_classification:
                return True
            classification = self._find_field_value(
                item, ["classification", "label", "category"]
            )
            return bool(
                classification
                and str(classification).lower() == filter_classification.lower()
            )

        # Convert data directly (no need to manage include_metadata since we always preserve all fields)
        result = self.convert_data_to_playground(
            data=classification_data, filter_function=classification_filter
        )

        return result

    def _normalize_input_data(
        self, data: Union[Dict[str, List[Dict]], List[Dict], Dict]
    ) -> List[Dict[str, Any]]:
        """
        Normalize various input data formats to a list of dictionaries with metadata.

        Args:
            data: Input data in various formats

        Returns:
            List of dictionaries with metadata about grouping
        """
        if isinstance(data, dict):
            # Check if it's grouped data (dict with lists as values) or single entry
            if any(isinstance(v, list) for v in data.values()):
                # Grouped data format
                normalized = []
                for group_key, items in data.items():
                    if isinstance(items, list):
                        for item in items:
                            item_with_meta = (
                                dict(item) if isinstance(item, dict) else {"data": item}
                            )
                            item_with_meta["_group"] = group_key
                            normalized.append(item_with_meta)
                    else:
                        # Single item in group
                        item_with_meta = (
                            dict(items) if isinstance(items, dict) else {"data": items}
                        )
                        item_with_meta["_group"] = group_key
                        normalized.append(item_with_meta)
                return normalized
            else:
                # Single dictionary entry
                return [data]
        elif isinstance(data, list):
            # List of dictionaries
            return [
                dict(item) if isinstance(item, dict) else {"data": item}
                for item in data
            ]
        else:
            # Single item
            return [{"data": data}]

    def _extract_field_value(
        self, data: Dict[str, Any], field_name: Optional[str]
    ) -> Optional[Any]:
        """Extract field value from data dictionary."""
        if not field_name:
            return None
        return data.get(field_name)

    def _find_field_value(
        self, data: Dict[str, Any], possible_fields: List[str]
    ) -> Optional[Any]:
        """Find field value from data trying multiple possible field names."""
        for field_name in possible_fields:
            if field_name in data and data[field_name] is not None:
                return data[field_name]
        return None

    def _extract_utterance_text(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract utterance text from the configured utterance field."""
        utterance = self._find_field_value(data, [self.field_mapping.utterance_field])
        if utterance:
            return str(utterance).strip()

        # If no utterance field found, try common alternatives
        alternatives = ["utterance", "text", "content", "query", "input", "data"]
        utterance = self._find_field_value(data, alternatives)
        if utterance:
            return str(utterance).strip()

        return None

    def _create_playground_entry_from_data(
        self, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create playground entry from normalized data - playground only needs utterance."""
        # Extract utterance text (the only required field for playground processing)
        utterance_text = self._extract_utterance_text(data)
        if not utterance_text:
            logger.warning("Skipping entry with no utterance text")
            return None

        # Build parameters dict starting with the required utterance
        parameters = {"utterance": utterance_text}

        # Add ALL other fields as preserved metadata (so we can restore original data later)
        excluded_utterance_fields = {self.field_mapping.utterance_field}
        excluded_utterance_fields.add("_group")  # Internal field

        for key, value in data.items():
            # Skip utterance fields we already used and internal fields
            if key not in excluded_utterance_fields and not key.startswith("_"):
                if value is not None and value != "" and value != []:
                    parameters[key] = value

        # Create simple playground INPUT structure
        playground_entry = {
            "input": {"parameters": json.dumps(parameters, ensure_ascii=False)}
        }

        return playground_entry

    def convert_playground_entries_to_playground(
        self, playground_entries: List[PlaygroundEntry]
    ) -> List[Dict[str, Any]]:
        """
        Convert PlaygroundEntry objects to playground format.

        Args:
            playground_entries: List of PlaygroundEntry objects

        Returns:
            List of playground-formatted entries
        """
        playground_data = []

        for entry in playground_entries:
            playground_item = self._create_playground_entry_from_dataclass(entry)
            playground_data.append(playground_item)

        return playground_data

    def _create_playground_entry_from_dataclass(
        self, entry: PlaygroundEntry
    ) -> Dict[str, Any]:
        """Create playground entry from PlaygroundEntry dataclass - playground only needs utterance."""

        # Build parameters dict starting with the required utterance
        parameters = {"utterance": entry.utterance}

        # Add segment if available in metadata and not default
        segment = entry.metadata.get("segment", "default")
        if segment != "default":
            parameters["segment"] = segment

        # Add source if available in metadata and not default
        source = entry.metadata.get("source", "general-data")
        if source != "general-data":
            parameters["source"] = source

        # Add all metadata fields as preserved data
        for key, value in entry.metadata.items():
            if (
                key not in ["segment", "source"]
                and value is not None
                and value != ""
                and value != []
            ):
                parameters[key] = value

        # Create simple playground INPUT structure
        playground_entry = {
            "input": {"parameters": json.dumps(parameters, ensure_ascii=False)}
        }

        return playground_entry

    def save_playground_data(
        self, playground_data: List[Dict[str, Any]], output_path: str, indent: int = 2
    ) -> None:
        """
        Save playground data to JSON file.

        Args:
            playground_data: List of playground-formatted entries
            output_path: Path to save the JSON file
            indent: JSON indentation level
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(playground_data, f, indent=indent, ensure_ascii=False)
            logger.info(
                f"Saved {len(playground_data)} playground entries to: {output_path}"
            )
        except Exception as e:
            logger.error(f"Error saving playground data to {output_path}: {e}")
            raise


def create_playground_converter(
    default_source: str = "general-data",
    default_results: Optional[List[Dict]] = None,
    field_mapping: Optional[FieldMapping] = None,
    **kwargs,
) -> PlaygroundDataConverter:
    """
    Factory function to create a PlaygroundDataConverter instance.

    Args:
        default_source: Default source identifier
        default_results: Default results structure (deprecated - use config)
        field_mapping: Custom field mapping configuration
        **kwargs: Additional configuration options

    Returns:
        Configured PlaygroundDataConverter instance
    """
    # Create conversion config
    config = ConversionConfig(default_source=default_source, **kwargs)

    # Create field mapping if not provided
    if field_mapping is None:
        field_mapping = FieldMapping()

    return PlaygroundDataConverter(
        field_mapping=field_mapping, conversion_config=config
    )


def create_classification_converter(
    default_source: str = "classification-data",
    **kwargs,
) -> PlaygroundDataConverter:
    """
    Factory function to create a converter optimized for classification data.

    Args:
        default_source: Default source identifier
        **kwargs: Additional configuration options

    Returns:
        Configured PlaygroundDataConverter for classification data
    """
    # Field mapping optimized for classification data
    field_mapping = FieldMapping(
        utterance_field="utterance"  # Default to "utterance" field
    )

    config = ConversionConfig(default_source=default_source, **kwargs)

    return PlaygroundDataConverter(
        field_mapping=field_mapping, conversion_config=config
    )


# Utility functions for common operations
def convert_classification_to_playground(
    classification_data: Dict[str, List[Dict]],
    output_path: str,
    filter_classification: Optional[str] = None,
    source: str = "classification-data",
    include_metadata: bool = True,
) -> int:
    """
    Convenience function to convert classification data to playground format and save it.

    Args:
        classification_data: Classification data dictionary
        output_path: Path to save playground JSON file
        filter_classification: Optional classification filter
        source: Source identifier for entries
        include_metadata: Whether to include metadata in results (ignored - always preserved)

    Returns:
        Number of entries converted
    """
    converter = create_classification_converter(default_source=source)

    playground_data = converter.convert_classification_data_to_playground(
        classification_data=classification_data,
        filter_classification=filter_classification,
        include_metadata=include_metadata,
    )

    converter.save_playground_data(playground_data, output_path)
    return len(playground_data)


def convert_general_data_to_playground(
    data: Union[Dict[str, List[Dict]], List[Dict], Dict],
    output_path: str,
    field_mapping: Optional[FieldMapping] = None,
    conversion_config: Optional[ConversionConfig] = None,
    filter_function: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> int:
    """
    Convenience function to convert general data to playground format and save it.

    Args:
        data: Input data in various formats
        output_path: Path to save playground JSON file
        field_mapping: Custom field mapping
        conversion_config: Custom conversion configuration
        filter_function: Optional filter function

    Returns:
        Number of entries converted
    """
    converter = PlaygroundDataConverter(
        field_mapping=field_mapping, conversion_config=conversion_config
    )

    playground_data = converter.convert_data_to_playground(
        data=data, filter_function=filter_function
    )

    converter.save_playground_data(playground_data, output_path)
    return len(playground_data)
