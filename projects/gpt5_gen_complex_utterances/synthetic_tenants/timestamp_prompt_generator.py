#!/usr/bin/env python3
"""
Timestamp Addition Prompt Generator

This module generates prompts for adding timestamps to utterances based on email accessibility.
For each utterance file (from split_results), it extracts the corresponding email account,
retrieves all accessible emails, and creates a prompt for an LLM to add appropriate timestamps.

The timestamp represents when the utterance would be asked to BizChat Copilot, ensuring
that all relevant emails needed to answer the utterance are accessible (i.e., sent before
that timestamp).

Key Features:
- Processes all email_*.tsv files from split_results directory
- Extracts email account from filename (e.g., alex.johnson from email_alex.johnson_utterances.tsv)
- Retrieves all emails accessible by that account using existing email extraction logic
- Generates comprehensive prompts with utterances and email context
- Creates timestamped output prompts for LLM processing
- Processes remaining untimestamped utterances from utterances_email_untimestamped.tsv

Usage:
    python timestamp_prompt_generator.py generate_timestamp_prompts
    python timestamp_prompt_generator.py generate_timestamp_prompts --split_results_dir="custom_split_dir"
    python timestamp_prompt_generator.py generate_timestamp_prompts --output_dir="custom_prompts"
    python timestamp_prompt_generator.py generate_remaining_timestamp_prompts
    python timestamp_prompt_generator.py generate_remaining_timestamp_prompts --untimestamped_file="custom_file.tsv"
"""

import csv
import os
import json
import logging
import fire
from typing import Dict, List, Optional
from datetime import datetime
from email_extractor_individual_accounts import IndividualEmailExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TimestampPromptGenerator:
    """Generate prompts for adding timestamps to utterances based on email accessibility"""

    def __init__(
        self,
        emails_file: str,
        users_file: str,
    ):
        """Initialize the timestamp prompt generator

        Args:
            emails_file: Path to the emails database JSON file
            users_file: Path to the users directory JSON file
        """
        self.emails_file = emails_file
        self.users_file = users_file
        self.email_extractor = IndividualEmailExtractor(emails_file, users_file)

    def _extract_email_account_from_filename(self, filename: str) -> Optional[str]:
        """Extract email account from filename like email_alex.johnson_utterances.tsv

        Args:
            filename: The TSV filename

        Returns:
            Email account string or None if extraction fails
        """
        if not filename.startswith("email_") or not filename.endswith(
            "_utterances.tsv"
        ):
            return None

        # Extract the middle part: email_alex.johnson_utterances.tsv -> alex.johnson
        account_part = filename[6:]  # Remove "email_" prefix
        account_part = account_part[:-15]  # Remove "_utterances.tsv" suffix

        return account_part if account_part else None

    def _extract_email_account_from_user_id(self, user_id: str) -> Optional[str]:
        """Extract email account from user_id by removing @ and everything after

        Args:
            user_id: The user_id string (e.g., troy.davis@VertexEdgeLabs@SyntheticTenant)

        Returns:
            Email account string or None if extraction fails
        """
        if not user_id or "@" not in user_id:
            return None

        # Extract part before first @
        email_account = user_id.split("@")[0]
        return email_account if email_account else None

    def _transform_utterance_to_output_format(self, utterance: Dict) -> Dict:
        """Transform utterance from input TSV format to expected output format

        Args:
            utterance: Original utterance dictionary from input TSV

        Returns:
            Transformed utterance dictionary with expected column names
        """
        # Extract email account from user_id
        user_id = utterance.get("user_id", "")
        email_account = user_id  # Keep full user_id as Email_Account

        # Transform to expected format
        transformed = {
            "Utterance": utterance.get("Utterance", ""),
            "Segment": utterance.get("Segment", ""),
            "Email_Account": email_account,
            "Source": "synthetic",  # Set to "synthetic" for all utterances as requested
        }

        return transformed

    def _read_utterances_from_file(self, filepath: str) -> List[Dict]:
        """Read utterances from TSV file

        Args:
            filepath: Path to the TSV file

        Returns:
            List of utterance dictionaries
        """
        utterances = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    utterances.append(dict(row))
            logger.info(f"📄 Read {len(utterances)} utterances from {filepath}")
            return utterances
        except Exception as e:
            logger.error(f"❌ Failed to read utterances from {filepath}: {str(e)}")
            return []

    def _format_email_summary(self, emails: List[Dict], max_emails: int = 10) -> str:
        """Create a summary of emails for the prompt

        Args:
            emails: List of email dictionaries
            max_emails: Maximum number of emails to include in summary

        Returns:
            Formatted email summary string
        """
        if not emails:
            return "No emails found"

        summary_lines = []
        for i, email in enumerate(emails[:max_emails]):
            timestamp = email.get("timestamp", "Unknown")
            subject = email.get("subject", "No subject")
            sender = email.get("sender", "Unknown sender")

            # Truncate long subjects
            if len(subject) > 80:
                subject = subject[:77] + "..."

            summary_lines.append(f"  {i+1}. {timestamp} - From: {sender}")
            summary_lines.append(f"     Subject: {subject}")

        if len(emails) > max_emails:
            summary_lines.append(f"  ... and {len(emails) - max_emails} more emails")

        return "\n".join(summary_lines)

    def _format_full_utterances_list(self, utterances: List[Dict]) -> str:
        """Format full utterances list for the prompt

        Args:
            utterances: List of utterance dictionaries

        Returns:
            Formatted full utterances list
        """
        if not utterances:
            return "No utterances found"

        lines = []
        for i, utterance in enumerate(utterances, 1):
            utterance_text = utterance.get("Utterance", "")
            segment = utterance.get("Segment", "")
            email_account = utterance.get("Email_Account", "")
            source = utterance.get("Source", "")

            lines.append(f"**{i}.** {utterance_text}")
            lines.append(f"   - Segment: {segment}")
            lines.append(f"   - Email Account: {email_account}")
            lines.append(f"   - Source: {source}")
            lines.append("")  # Empty line for spacing

        return "\n".join(lines)

    def generate_prompt_for_account(
        self,
        email_account: str,
        utterances: List[Dict],
        output_file: str,
        template_file: str = "prompt_add_timestamp.md",
    ) -> None:
        """Generate timestamp addition prompt for a specific email account

        Args:
            email_account: Email account to process
            utterances: List of utterances for this account
            output_file: Path to save the generated prompt
            template_file: Path to the prompt template file
        """
        logger.info(f"🚀 Generating timestamp prompt for account: {email_account}")

        # Read the prompt template (fail if it doesn't exist)
        try:
            with open(template_file, "r", encoding="utf-8") as f:
                template_content = f.read()
        except FileNotFoundError:
            logger.error(f"❌ Template file not found: {template_file}")
            raise FileNotFoundError(
                f"Required template file not found: {template_file}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to read template {template_file}: {str(e)}")
            raise

        # Extract emails for the account
        try:
            email_data = self.email_extractor.extract_emails_for_account(
                target_account=email_account,
                output_file=None,  # Don't save intermediate files
            )
        except Exception as e:
            logger.error(f"❌ Failed to extract emails for {email_account}: {str(e)}")
            return

        # Prepare email summaries
        sent_emails = email_data.get("sent_emails", [])
        received_emails = email_data.get("received_emails", [])

        sent_summary = self._format_email_summary(sent_emails)
        received_summary = self._format_email_summary(received_emails)

        # Prepare full utterances list
        full_utterances_list = self._format_full_utterances_list(utterances)

        # Prepare email JSON data (simplified for prompt)
        email_json_simplified = {
            "statistics": email_data.get("statistics", {}),
            "sent_emails": sent_emails,
            "received_emails": received_emails,
        }
        email_json_data = json.dumps(
            email_json_simplified, indent=2, ensure_ascii=False
        )

        # Fill in the template
        filled_prompt = template_content.replace("{email_account}", email_account)
        filled_prompt = filled_prompt.replace("{sent_count}", str(len(sent_emails)))
        filled_prompt = filled_prompt.replace(
            "{received_count}", str(len(received_emails))
        )
        filled_prompt = filled_prompt.replace("{sent_emails_summary}", sent_summary)
        filled_prompt = filled_prompt.replace(
            "{received_emails_summary}", received_summary
        )
        filled_prompt = filled_prompt.replace("{email_json_data}", email_json_data)
        filled_prompt = filled_prompt.replace(
            "{full_utterances_list}", full_utterances_list
        )
        filled_prompt = filled_prompt.replace("{utterance_count}", str(len(utterances)))

        # Save the generated prompt
        try:
            # Create directory only if there's a directory part
            output_dir = os.path.dirname(output_file)
            if output_dir:  # Only create if there's a directory part
                os.makedirs(output_dir, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(filled_prompt)

            logger.info(f"📝 Generated prompt: {output_file}")
            logger.info(f"📧 Email account: {email_account}")
            logger.info(f"📊 Utterances: {len(utterances)}")
            logger.info(f"📤 Sent emails: {len(sent_emails)}")
            logger.info(f"📥 Received emails: {len(received_emails)}")

        except Exception as e:
            logger.error(f"❌ Failed to save prompt to {output_file}: {str(e)}")
            raise

    def generate_timestamp_prompts(
        self,
        split_results_dir: str = "split_results",
        output_dir: str = "timestamp_prompts",
        template_file: str = "prompt_add_timestamp.md",
    ) -> None:
        """Generate timestamp addition prompts for all email accounts

        Args:
            split_results_dir: Directory containing split TSV files
            output_dir: Directory to save generated prompts
            template_file: Path to the prompt template file
        """
        logger.info(f"🚀 Starting timestamp prompt generation...")
        logger.info(f"📁 Split results directory: {split_results_dir}")
        logger.info(f"📁 Output directory: {output_dir}")

        if not os.path.exists(split_results_dir):
            raise FileNotFoundError(
                f"✗ Split results directory not found: {split_results_dir}"
            )

        # Find all email_*.tsv files
        email_files = []
        for filename in os.listdir(split_results_dir):
            if filename.startswith("email_") and filename.endswith("_utterances.tsv"):
                email_files.append(filename)

        if not email_files:
            logger.warning(
                f"⚠️ No email_*_utterances.tsv files found in {split_results_dir}"
            )
            return

        logger.info(f"📄 Found {len(email_files)} email utterance files to process")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process each file
        processed_count = 0
        failed_count = 0

        for filename in sorted(email_files):
            try:
                # Extract email account from filename
                email_account = self._extract_email_account_from_filename(filename)
                if not email_account:
                    logger.warning(
                        f"⚠️ Could not extract email account from filename: {filename}"
                    )
                    failed_count += 1
                    continue

                # Read utterances from file
                filepath = os.path.join(split_results_dir, filename)
                utterances = self._read_utterances_from_file(filepath)

                if not utterances:
                    logger.warning(f"⚠️ No utterances found in file: {filename}")
                    failed_count += 1
                    continue

                # Generate output filename
                output_filename = f"timestamp_prompt_{email_account}.md"
                output_path = os.path.join(output_dir, output_filename)

                # Generate the prompt
                self.generate_prompt_for_account(
                    email_account=email_account,
                    utterances=utterances,
                    output_file=output_path,
                    template_file=template_file,
                )

                processed_count += 1

            except Exception as e:
                logger.error(f"❌ Failed to process {filename}: {str(e)}")
                failed_count += 1

        # Summary
        logger.info(f"✅ Timestamp prompt generation completed!")
        logger.info(f"📊 Successfully processed: {processed_count} files")
        logger.info(f"❌ Failed to process: {failed_count} files")
        logger.info(f"📁 Prompts saved to: {output_dir}")

        if processed_count > 0:
            logger.info(f"\n💡 Next steps:")
            logger.info(f"1. Review the generated prompts in {output_dir}/")
            logger.info(
                f"2. Send each prompt to an LLM to add timestamps to utterances"
            )
            logger.info(
                f"3. Use the timestamped utterances for BizChat Copilot testing"
            )

    def generate_remaining_timestamp_prompts(
        self,
        untimestamped_file: str = "results/utterances_email_untimestamped.tsv",
        output_dir: str = "results/timestamp_prompts_remaining",
        template_file: str = "prompt_add_timestamp.md",
    ) -> None:
        """Generate timestamp addition prompts for remaining untimestamped utterances

        Args:
            untimestamped_file: Path to the TSV file containing untimestamped utterances
            output_dir: Directory to save generated prompts
            template_file: Path to the prompt template file
        """
        logger.info(f"🚀 Starting remaining timestamp prompt generation...")
        logger.info(f"📄 Untimestamped file: {untimestamped_file}")
        logger.info(f"📁 Output directory: {output_dir}")

        if not os.path.exists(untimestamped_file):
            raise FileNotFoundError(
                f"✗ Untimestamped utterances file not found: {untimestamped_file}"
            )

        # Read all untimestamped utterances
        utterances = self._read_utterances_from_file(untimestamped_file)
        if not utterances:
            logger.warning(f"⚠️ No utterances found in file: {untimestamped_file}")
            return

        # Filter for email segment utterances only
        email_utterances = [
            utterance
            for utterance in utterances
            if utterance.get("Segment", "").lower() == "email"
        ]

        if not email_utterances:
            logger.warning(f"⚠️ No email utterances found in file: {untimestamped_file}")
            return

        logger.info(f"📧 Found {len(email_utterances)} email utterances to process")

        # Group utterances by email account and transform to output format
        utterances_by_account = {}
        for utterance in email_utterances:
            user_id = utterance.get("user_id", "")
            email_account = self._extract_email_account_from_user_id(user_id)

            if not email_account:
                logger.warning(
                    f"⚠️ Could not extract email account from user_id: {user_id}"
                )
                continue

            # Transform utterance to expected output format
            transformed_utterance = self._transform_utterance_to_output_format(
                utterance
            )

            if email_account not in utterances_by_account:
                utterances_by_account[email_account] = []
            utterances_by_account[email_account].append(transformed_utterance)

        if not utterances_by_account:
            logger.warning(f"⚠️ No valid email accounts found in utterances")
            return

        logger.info(f"👥 Found {len(utterances_by_account)} unique email accounts")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process each email account
        processed_count = 0
        failed_count = 0

        for email_account, account_utterances in sorted(utterances_by_account.items()):
            try:
                logger.info(
                    f"📧 Processing {email_account}: {len(account_utterances)} utterances"
                )

                # Generate output filename
                output_filename = f"remaining_timestamp_prompt_{email_account}.md"
                output_path = os.path.join(output_dir, output_filename)

                # Generate the prompt
                self.generate_prompt_for_account(
                    email_account=email_account,
                    utterances=account_utterances,
                    output_file=output_path,
                    template_file=template_file,
                )

                processed_count += 1

            except Exception as e:
                logger.error(f"❌ Failed to process {email_account}: {str(e)}")
                failed_count += 1

        # Summary
        logger.info(f"✅ Remaining timestamp prompt generation completed!")
        logger.info(f"📊 Successfully processed: {processed_count} email accounts")
        logger.info(f"❌ Failed to process: {failed_count} email accounts")
        logger.info(f"📁 Prompts saved to: {output_dir}")

        if processed_count > 0:
            logger.info(f"\n💡 Next steps:")
            logger.info(f"1. Review the generated prompts in {output_dir}/")
            logger.info(
                f"2. Send each prompt to an LLM to add timestamps to utterances"
            )
            logger.info(
                f"3. Use the timestamped utterances for BizChat Copilot testing"
            )


def generate_timestamp_prompts(
    split_results_dir: str = "results/split_results",
    output_dir: str = "results/timestamp_prompts",
    emails_file: str = "data/emails.config.json",
    users_file: str = "data/users.config.json",
    template_file: str = "prompt_add_timestamp.md",
) -> None:
    """Generate timestamp addition prompts for all email accounts

    Args:
        split_results_dir: Directory containing split TSV files (default: "split_results")
        output_dir: Directory to save generated prompts (default: "timestamp_prompts")
        emails_file: Path to the emails database JSON file
        users_file: Path to the users directory JSON file
        template_file: Path to the prompt template file
    """
    generator = TimestampPromptGenerator(emails_file=emails_file, users_file=users_file)
    generator.generate_timestamp_prompts(
        split_results_dir=split_results_dir,
        output_dir=output_dir,
        template_file=template_file,
    )


def generate_remaining_timestamp_prompts(
    untimestamped_file: str = "results/utterances_email_untimestamped.tsv",
    output_dir: str = "results/timestamp_prompts_remaining",
    emails_file: str = "data/emails.config.json",
    users_file: str = "data/users.config.json",
    template_file: str = "prompt_add_timestamp.md",
) -> None:
    """Generate timestamp addition prompts for remaining untimestamped utterances

    Args:
        untimestamped_file: Path to the TSV file containing untimestamped utterances
        output_dir: Directory to save generated prompts
        emails_file: Path to the emails database JSON file
        users_file: Path to the users directory JSON file
        template_file: Path to the prompt template file
    """
    generator = TimestampPromptGenerator(emails_file=emails_file, users_file=users_file)
    generator.generate_remaining_timestamp_prompts(
        untimestamped_file=untimestamped_file,
        output_dir=output_dir,
        template_file=template_file,
    )


def generate_prompt_for_account(
    email_account: str,
    utterances_input: str,
    output_file: str,
    emails_file: str = "data/emails.config.json",
    users_file: str = "data/users.config.json",
    template_file: str = "prompt_add_timestamp.md",
) -> None:
    """Generate timestamp addition prompt for a specific email account

    Args:
        email_account: Email account to process
        utterances_input: JSON string containing utterances data OR path to JSON file
        output_file: Path to save the generated prompt
        emails_file: Path to the emails database JSON file
        users_file: Path to the users directory JSON file
        template_file: Path to the prompt template file
    """

    # Parse the utterances - handle both JSON string and file path
    try:
        # Check if it's a file path (has .json extension or exists as file)
        if utterances_input.endswith(".json") or (
            os.path.exists(utterances_input) and os.path.isfile(utterances_input)
        ):
            with open(utterances_input, "r", encoding="utf-8") as f:
                utterances = json.load(f)
            logger.info(f"📁 Loaded utterances from file: {utterances_input}")
        else:
            # Assume it's a JSON string
            utterances = json.loads(utterances_input)
            logger.info("📝 Parsed utterances from JSON string")
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON format for utterances: {str(e)}")
        raise
    except FileNotFoundError as e:
        logger.error(f"❌ Utterances file not found: {utterances_input}")
        raise

    generator = TimestampPromptGenerator(emails_file=emails_file, users_file=users_file)
    generator.generate_prompt_for_account(
        email_account=email_account,
        utterances=utterances,
        output_file=output_file,
        template_file=template_file,
    )


if __name__ == "__main__":
    fire.Fire()
