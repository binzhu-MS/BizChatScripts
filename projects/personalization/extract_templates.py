"""
Extract personalization templates from a personalized set of utterances (such as 1K Golden set).
"""

import csv
import re
import logging
import fire

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEMPLATE_PATTERN = re.compile(r"<[^>]+>")


def extract_templates(input_file, output_file, column=1):
    """
    Extract rows with templates containing angle brackets from a TSV file.
    Args:
        input_file: Path to the input TSV file
        output_file: Path to the output TSV file
        column: Column index (0-based) to check for templates (default: 1)
    """
    logger.info(f"Reading from {input_file}")
    entities = set()
    replaced_entities = set()
    always_unreplaced_entities = set()

    # Track which entities have been seen and whether they've been replaced at least once
    entity_replacement_status = {}  # entity -> has_been_replaced_at_least_once

    with open(input_file, encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")
        header = next(reader)
        writer.writerow(header)
        count = 0

        for row in reader:
            if len(row) > column and TEMPLATE_PATTERN.search(row[column]):
                writer.writerow(row)
                found = TEMPLATE_PATTERN.findall(row[column])
                entities.update(found)

                # Track replacement status for each entity
                for entity in found:
                    if entity not in entity_replacement_status:
                        entity_replacement_status[entity] = False

                    # Check if this entity is replaced in the first column for this row
                    if (
                        len(row) > 0 and entity not in row[0]
                    ):  # Entity NOT in first column = replaced
                        entity_replacement_status[entity] = True

                count += 1

    # Categorize entities based on whether they've been replaced at least once
    for entity, has_been_replaced in entity_replacement_status.items():
        if has_been_replaced:
            replaced_entities.add(entity)
        else:
            always_unreplaced_entities.add(entity)

    logger.info(f"Extracted {count} templates written to {output_file}")

    if entities:
        logger.info(f"{len(entities)} entities found in column {column}:")
        for ent in sorted(entities):
            print(ent)

    print("\n=== ENTITY REPLACEMENT ANALYSIS ===")
    if replaced_entities:
        print(f"\n1. REPLACED ENTITIES ({len(replaced_entities)} total):")
        print("   These entities have been replaced with actual values at least once:")
        for ent in sorted(replaced_entities):
            print(f"   {ent}")

    if always_unreplaced_entities:
        print(
            f"\n2. ALWAYS UNREPLACED ENTITIES ({len(always_unreplaced_entities)} total):"
        )
        print("   These entities are ALWAYS kept as placeholders (never replaced):")
        for ent in sorted(always_unreplaced_entities):
            print(f"   {ent}")

    if not replaced_entities and not always_unreplaced_entities:
        print("   No entities found for analysis.")


if __name__ == "__main__":
    fire.Fire(extract_templates)
