# check_seval_fields.py
#
# Copyright (c) Microsoft Corporation. All rights reserved.

"""
Check available fields per entity type in raw SEVAL data.
"""

import json
from pathlib import Path
from collections import defaultdict


def main():
    """Check all fields available in raw SEVAL data."""
    data_dir = Path("seval_data/144683_scraping_raw_data_output")
    files = list(data_dir.glob("control_*.json"))[:50]
    
    # Track all fields per type
    type_fields = defaultdict(set)
    # Track all fields in source object per type
    type_source_fields = defaultdict(set)
    
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        msgs = data.get("requests", [{}])[0].get("response_body", {}).get("messages", [])
        eval_msgs = [m for m in msgs if m.get("messageType") == "EvaluationData"]
        if not eval_msgs:
            continue
            
        turn_data = eval_msgs[0].get("evaluationData", {}).get("turnData", [])
        if not turn_data:
            continue
            
        for it in turn_data[-1].get("orchestrationIterations", []):
            for action in it.get("modelActions", []):
                for inv in action.get("toolInvocations", []):
                    processed = inv.get("processedResult")
                    if not processed:
                        continue
                    try:
                        pr = json.loads(processed)
                    except Exception:
                        continue
                    for r in pr.get("results", [])[:5]:
                        entity_type = r.get("type", "Unknown")
                        type_fields[entity_type].update(r.keys())
                        
                        # Check source field
                        if "source" in r:
                            source = r["source"]
                            if isinstance(source, dict):
                                type_source_fields[entity_type].update(source.keys())
    
    print("=" * 60)
    print("FIELDS PER ENTITY TYPE (top level)")
    print("=" * 60)
    for t, fs in sorted(type_fields.items()):
        print(f"\n{t}:")
        for field in sorted(fs):
            print(f"  - {field}")
    
    print("\n" + "=" * 60)
    print("SOURCE OBJECT FIELDS PER ENTITY TYPE")
    print("=" * 60)
    for t, fs in sorted(type_source_fields.items()):
        print(f"\n{t}:")
        for field in sorted(fs):
            print(f"  - {field}")


if __name__ == "__main__":
    main()
