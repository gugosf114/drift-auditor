"""
Batch Results Validator
=======================
Reads batch audit results and generates a verifiable summary.

Outputs batch_summary.json with:
- Total conversations per platform
- Total messages processed
- Aggregate metrics matching README claims
- SHA-256 hash of each results file for integrity

Run: python batch_validate.py
Outputs: batch_summary.json (commit this to the repo)

Author: George Abrahamyants + Claude (Opus 4.6)
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime


def validate_batch_results():
    """
    Reads batch audit results and generates a verifiable summary.

    Outputs batch_summary.json with:
    - Total conversations per platform
    - Total messages processed
    - Aggregate metrics matching README claims
    - SHA-256 hash of each results file for integrity
    """
    summary = {
        "generated_at": datetime.now().isoformat(),
        "platforms": {},
        "totals": {},
        "file_hashes": {},
    }

    # Claude results
    claude_dir = Path("batch_results")
    chatgpt_dir = Path("batch_results_chatgpt")

    for platform, result_dir, result_file in [
        ("claude", claude_dir, "batch_results.json"),
        ("chatgpt", chatgpt_dir, "batch_results_chatgpt.json"),
    ]:
        filepath = result_dir / result_file
        if not filepath.exists():
            # Try finding any JSON file in the directory
            json_files = list(result_dir.glob("*.json")) if result_dir.exists() else []
            if json_files:
                filepath = json_files[0]
            else:
                summary["platforms"][platform] = {
                    "error": f"No results file found in {result_dir}"
                }
                continue

        # Hash the file for integrity verification
        with open(filepath, "rb") as f:
            file_bytes = f.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
        summary["file_hashes"][str(filepath)] = file_hash

        # Parse results
        data = json.loads(file_bytes.decode("utf-8"))

        if isinstance(data, list):
            results = data
        elif isinstance(data, dict):
            # Might be wrapped in a key
            results = data.get("results", data.get("audits", [data]))
            if not isinstance(results, list):
                results = [results]
        else:
            summary["platforms"][platform] = {"error": "Unrecognized format"}
            continue

        conversation_count = len(results)
        total_messages = 0
        drift_scores = []
        correction_counts = []
        omission_counts = []

        for result in results:
            turns = result.get("total_turns", 0)
            total_messages += turns

            scores = result.get("scores", result.get("summary_scores", {}))
            if scores:
                drift_scores.append(scores.get("overall_drift_score", 0))
                correction_counts.append(scores.get("correction_events_total", 0))
                omission_counts.append(scores.get("omission_flag_count", 0))

        avg_drift = (
            sum(drift_scores) / len(drift_scores) if drift_scores else 0
        )
        avg_corrections = (
            sum(correction_counts) / len(correction_counts)
            if correction_counts
            else 0
        )

        summary["platforms"][platform] = {
            "conversations": conversation_count,
            "total_messages": total_messages,
            "avg_drift_score": round(avg_drift, 1),
            "avg_corrections_per_conversation": round(avg_corrections, 1),
            "drift_score_distribution": {
                "1-3 (low)": sum(1 for s in drift_scores if 1 <= s <= 3),
                "4-6 (medium)": sum(1 for s in drift_scores if 4 <= s <= 6),
                "7-10 (high)": sum(1 for s in drift_scores if 7 <= s <= 10),
            },
        }

    # Totals
    total_convos = sum(
        p.get("conversations", 0)
        for p in summary["platforms"].values()
        if isinstance(p, dict) and "conversations" in p
    )
    total_msgs = sum(
        p.get("total_messages", 0)
        for p in summary["platforms"].values()
        if isinstance(p, dict) and "total_messages" in p
    )
    summary["totals"] = {
        "total_conversations": total_convos,
        "total_messages": total_msgs,
        "readme_claims_match": {
            "conversations_512": total_convos == 512,
            "messages_37536": total_msgs == 37536,
        },
    }

    # Write summary
    output_path = "batch_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {output_path}")
    print(f"Total conversations: {total_convos} (README claims 512)")
    print(f"Total messages: {total_msgs} (README claims 37,536)")
    print(f"Match: {total_convos == 512 and total_msgs == 37536}")

    return summary


if __name__ == "__main__":
    validate_batch_results()
