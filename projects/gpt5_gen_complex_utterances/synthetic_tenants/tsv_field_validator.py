#!/usr/bin/env python3
"""
TSV Field Validator

This module validates TSV files to check for missing (empty) fields, ensuring data completeness
and quality. It provides detailed reports on field coverage, identifies problematic rows,
and generates comprehensive statistics.

Key Features:
- Validates all fields in TSV files for completeness
- Identifies rows with missing or empty fields
- Generates detailed validation reports
- Supports custom field validation rules
- Provides statistics on field coverage
- Exports validation results to various formats

Usage:
    python tsv_field_validator.py validate_file "results/utterances_seval_timestamped.tsv"
    python tsv_field_validator.py validate_file "file.tsv" --output_report="validation_report.json"
    python tsv_field_validator.py validate_file "file.tsv" --show_details=True --max_issues=10
"""

import csv
import os
import json
import logging
import fire
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class FieldIssue:
    """Represents a field validation issue"""

    row_number: int
    field_name: str
    field_value: str
    issue_type: str
    severity: str


@dataclass
class ValidationResult:
    """Represents the complete validation result"""

    file_path: str
    total_rows: int
    total_fields: int
    field_names: List[str]
    issues: List[FieldIssue]
    field_coverage: Dict[str, Dict[str, int]]
    validation_timestamp: str
    is_valid: bool


class TSVFieldValidator:
    """Validates TSV files for missing or empty fields"""

    def __init__(self):
        """Initialize the TSV field validator"""
        self.validation_rules = {
            "empty": self._check_empty,
            "whitespace_only": self._check_whitespace_only,
            "null_values": self._check_null_values,
        }

    def _check_empty(self, value: str) -> bool:
        """Check if value is empty"""
        return value == ""

    def _check_whitespace_only(self, value: str) -> bool:
        """Check if value contains only whitespace"""
        return value.strip() == ""

    def _check_null_values(self, value: str) -> bool:
        """Check if value represents null/none"""
        null_values = {"null", "none", "n/a", "na", "-", "undefined", "nil"}
        return value.lower().strip() in null_values

    def validate_file(
        self,
        file_path: str,
        output_report: Optional[str] = None,
        show_details: bool = False,
        max_issues: Optional[int] = None,
        validation_rules: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate a TSV file for missing fields

        Args:
            file_path: Path to the TSV file to validate
            output_report: Optional path to save validation report
            show_details: Whether to show detailed issue information
            max_issues: Maximum number of issues to display/store
            validation_rules: List of validation rules to apply

        Returns:
            ValidationResult object containing validation details
        """
        logger.info(f"🔍 Starting TSV field validation for: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Set default validation rules
        if validation_rules is None:
            validation_rules = ["empty", "whitespace_only"]

        # Read and validate the TSV file
        issues = []
        field_names = []
        total_rows = 0
        field_coverage = {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                field_names = list(reader.fieldnames or [])

                # Initialize field coverage tracking
                for field_name in field_names:
                    field_coverage[field_name] = {
                        "total": 0,
                        "valid": 0,
                        "empty": 0,
                        "whitespace_only": 0,
                        "null_values": 0,
                    }

                # Process each row
                for row_num, row in enumerate(
                    reader, start=2
                ):  # Start at 2 (header is row 1)
                    total_rows += 1

                    # Check each field in the row
                    for field_name in field_names:
                        field_value = row.get(field_name, "")
                        field_coverage[field_name]["total"] += 1

                        # Apply validation rules
                        field_valid = True
                        for rule_name in validation_rules:
                            if rule_name in self.validation_rules:
                                rule_func = self.validation_rules[rule_name]
                                if rule_func(field_value):
                                    # Field failed this rule
                                    issue = FieldIssue(
                                        row_number=row_num,
                                        field_name=field_name,
                                        field_value=field_value,
                                        issue_type=rule_name,
                                        severity=(
                                            "warning"
                                            if rule_name == "whitespace_only"
                                            else "error"
                                        ),
                                    )
                                    issues.append(issue)
                                    field_coverage[field_name][rule_name] += 1
                                    field_valid = False

                        if field_valid:
                            field_coverage[field_name]["valid"] += 1

                        # Stop if we've reached max issues
                        if max_issues and len(issues) >= max_issues:
                            break

                    if max_issues and len(issues) >= max_issues:
                        logger.warning(
                            f"⚠️ Reached maximum issue limit ({max_issues}), stopping validation"
                        )
                        break

        except Exception as e:
            logger.error(f"❌ Error reading TSV file: {str(e)}")
            raise

        # Create validation result
        validation_result = ValidationResult(
            file_path=file_path,
            total_rows=total_rows,
            total_fields=len(field_names),
            field_names=field_names,
            issues=issues,
            field_coverage=field_coverage,
            validation_timestamp=datetime.now().isoformat(),
            is_valid=len(issues) == 0,
        )

        # Display results
        self._display_validation_results(validation_result, show_details, max_issues)

        # Save report if requested
        if output_report:
            self._save_validation_report(validation_result, output_report)

        return validation_result

    def _display_validation_results(
        self, result: ValidationResult, show_details: bool, max_issues: Optional[int]
    ) -> None:
        """Display validation results to console"""
        logger.info(f"📊 Validation Results for: {result.file_path}")
        logger.info(f"📄 Total rows: {result.total_rows}")
        logger.info(f"📋 Total fields: {result.total_fields}")
        logger.info(f"🏷️ Field names: {', '.join(result.field_names)}")
        logger.info(f"❗ Total issues found: {len(result.issues)}")
        logger.info(f"✅ File is valid: {result.is_valid}")

        # Display field coverage summary
        logger.info(f"\n📈 Field Coverage Summary:")
        for field_name, coverage in result.field_coverage.items():
            total = coverage["total"]
            valid = coverage["valid"]
            empty = coverage["empty"]
            whitespace = coverage["whitespace_only"]
            null_vals = coverage["null_values"]

            logger.info(f"  📌 {field_name}:")
            logger.info(f"     ✅ Valid: {valid}/{total} ({(valid/total*100):.1f}%)")
            if empty > 0:
                logger.info(f"     🚫 Empty: {empty} ({(empty/total*100):.1f}%)")
            if whitespace > 0:
                logger.info(
                    f"     ⚪ Whitespace only: {whitespace} ({(whitespace/total*100):.1f}%)"
                )
            if null_vals > 0:
                logger.info(
                    f"     ❌ Null values: {null_vals} ({(null_vals/total*100):.1f}%)"
                )

        # Display detailed issues if requested
        if show_details and result.issues:
            logger.info(f"\n🔍 Detailed Issues (showing up to {max_issues or 'all'}):")
            display_issues = result.issues[:max_issues] if max_issues else result.issues

            for issue in display_issues:
                severity_icon = "🚨" if issue.severity == "error" else "⚠️"
                logger.info(
                    f"  {severity_icon} Row {issue.row_number}, Field '{issue.field_name}': "
                    f"{issue.issue_type} (Value: '{issue.field_value}')"
                )

            if max_issues and len(result.issues) > max_issues:
                logger.info(
                    f"     ... and {len(result.issues) - max_issues} more issues"
                )

        # Summary
        if result.is_valid:
            logger.info(f"\n🎉 All fields are properly filled! No issues detected.")
        else:
            error_count = sum(1 for issue in result.issues if issue.severity == "error")
            warning_count = len(result.issues) - error_count
            logger.info(f"\n⚠️ Found {error_count} errors and {warning_count} warnings")

    def _save_validation_report(
        self, result: ValidationResult, output_path: str
    ) -> None:
        """Save validation report to file"""
        try:
            # Convert result to dictionary for JSON serialization
            report_data = asdict(result)

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Validation report saved to: {output_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save validation report: {str(e)}")
            raise

    def validate_specific_fields(
        self,
        file_path: str,
        required_fields: List[str],
        output_report: Optional[str] = None,
    ) -> ValidationResult:
        """Validate specific fields in a TSV file

        Args:
            file_path: Path to the TSV file to validate
            required_fields: List of field names that must be present and non-empty
            output_report: Optional path to save validation report

        Returns:
            ValidationResult object containing validation details
        """
        logger.info(f"🔍 Validating specific fields: {', '.join(required_fields)}")

        # First, do general validation
        result = self.validate_file(file_path, show_details=False)

        # Check if all required fields exist
        missing_fields = [
            field for field in required_fields if field not in result.field_names
        ]
        if missing_fields:
            logger.error(f"❌ Missing required fields: {', '.join(missing_fields)}")

        # Filter issues to only required fields
        filtered_issues = [
            issue for issue in result.issues if issue.field_name in required_fields
        ]

        # Create new result with filtered issues
        filtered_result = ValidationResult(
            file_path=result.file_path,
            total_rows=result.total_rows,
            total_fields=len(required_fields),
            field_names=required_fields,
            issues=filtered_issues,
            field_coverage={
                field: result.field_coverage[field]
                for field in required_fields
                if field in result.field_coverage
            },
            validation_timestamp=result.validation_timestamp,
            is_valid=len(filtered_issues) == 0 and len(missing_fields) == 0,
        )

        # Display results
        self._display_validation_results(
            filtered_result, show_details=True, max_issues=None
        )

        # Save report if requested
        if output_report:
            self._save_validation_report(filtered_result, output_report)

        return filtered_result

    def quick_check(self, file_path: str) -> Tuple[bool, int, Dict[str, int]]:
        """Perform a quick validation check

        Args:
            file_path: Path to the TSV file to validate

        Returns:
            Tuple of (is_valid, total_issues, issues_by_type)
        """
        result = self.validate_file(file_path, show_details=False)

        issues_by_type = {}
        for issue in result.issues:
            issues_by_type[issue.issue_type] = (
                issues_by_type.get(issue.issue_type, 0) + 1
            )

        return result.is_valid, len(result.issues), issues_by_type


# Standalone functions for Fire CLI
def validate_file(
    file_path: str,
    output_report: Optional[str] = None,
    show_details: bool = False,
    max_issues: Optional[int] = None,
    validation_rules: Optional[str] = None,
) -> None:
    """Validate a TSV file for missing fields

    Args:
        file_path: Path to the TSV file to validate
        output_report: Optional path to save validation report
        show_details: Whether to show detailed issue information
        max_issues: Maximum number of issues to display/store
        validation_rules: Comma-separated string of validation rules
    """
    validator = TSVFieldValidator()

    # Parse validation rules
    rules = None
    if validation_rules:
        if isinstance(validation_rules, str):
            rules = [rule.strip() for rule in validation_rules.split(",")]
        else:
            rules = list(validation_rules)  # Handle if it's already a list/tuple

    validator.validate_file(
        file_path=file_path,
        output_report=output_report,
        show_details=show_details,
        max_issues=max_issues,
        validation_rules=rules,
    )


def validate_specific_fields(
    file_path: str,
    required_fields: str,
    output_report: Optional[str] = None,
) -> None:
    """Validate specific fields in a TSV file

    Args:
        file_path: Path to the TSV file to validate
        required_fields: Comma-separated list of required field names
        output_report: Optional path to save validation report
    """
    validator = TSVFieldValidator()

    # Parse required fields
    fields = [field.strip() for field in required_fields.split(",")]

    validator.validate_specific_fields(
        file_path=file_path,
        required_fields=fields,
        output_report=output_report,
    )


def quick_check(file_path: str) -> None:
    """Perform a quick validation check

    Args:
        file_path: Path to the TSV file to validate
    """
    validator = TSVFieldValidator()
    is_valid, total_issues, issues_by_type = validator.quick_check(file_path)

    logger.info(f"🔍 Quick Check Results for: {file_path}")
    logger.info(f"✅ File is valid: {is_valid}")
    logger.info(f"❗ Total issues: {total_issues}")

    if issues_by_type:
        logger.info("📊 Issues by type:")
        for issue_type, count in issues_by_type.items():
            logger.info(f"  - {issue_type}: {count}")


if __name__ == "__main__":
    fire.Fire()
