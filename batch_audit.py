"""
Batch Audit Runner — Process all conversations from Claude data export.
Produces a ranked summary of drift severity across the full dataset.
"""
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from drift_auditor import audit_conversation, AuditReport, format_report, report_to_json


def load_conversations(export_path: str) -> list[dict]:
    """Load conversations from Claude data export JSON."""
    with open(export_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def conversation_to_text(conv: dict) -> str:
    """Convert a Claude export conversation to the JSON format our parser handles."""
    messages = conv.get('chat_messages', [])
    if not messages:
        return ""
    return json.dumps(messages)


def run_batch(export_path: str, min_messages: int = 10, output_dir: str = "batch_results"):
    """Run drift audit on all conversations meeting minimum message threshold."""
    
    print(f"Loading conversations from {export_path}...")
    conversations = load_conversations(export_path)
    print(f"Found {len(conversations)} conversations")
    
    # Filter by minimum messages
    eligible = []
    for conv in conversations:
        msgs = conv.get('chat_messages', [])
        if len(msgs) >= min_messages:
            eligible.append(conv)
    
    print(f"{len(eligible)} conversations with >= {min_messages} messages")
    print(f"Skipping {len(conversations) - len(eligible)} short conversations")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    errors = []
    
    for i, conv in enumerate(eligible):
        name = conv.get('name', 'unnamed')[:60]
        uuid = conv.get('uuid', 'unknown')[:8]
        msg_count = len(conv.get('chat_messages', []))
        
        safe_name = name.encode('ascii', 'replace').decode('ascii')
        print(f"[{i+1}/{len(eligible)}] {safe_name} ({msg_count} msgs)...", end=" ", flush=True)
        
        try:
            raw_text = conversation_to_text(conv)
            if not raw_text:
                print("SKIP (empty)")
                continue
            
            report = audit_conversation(
                raw_text=raw_text,
                conversation_id=f"{uuid}_{name[:30]}",
                window_size=50,
                overlap=10,
            )
            
            scores = report.summary_scores
            overall = scores.get('overall_drift_score', 0)
            
            results.append({
                'uuid': conv.get('uuid', ''),
                'name': name,
                'created_at': conv.get('created_at', ''),
                'message_count': msg_count,
                'total_turns': report.total_turns,
                'instructions_extracted': report.instructions_extracted,
                'overall_score': overall,
                'commission_score': scores.get('commission_score', 0),
                'omission_score': scores.get('omission_score', 0),
                'persistence_score': scores.get('correction_persistence_score', 0),
                'barometer_score': scores.get('barometer_score', 0),
                'structural_score': scores.get('structural_score', 0),
                'commission_flags': scores.get('commission_flag_count', 0),
                'omission_flags': scores.get('omission_flag_count', 0),
                'corrections_total': scores.get('correction_events_total', 0),
                'corrections_failed': scores.get('corrections_failed', 0),
                'conflict_pairs': scores.get('conflict_pairs_count', 0),
                'void_events': scores.get('void_events_count', 0),
                'shadow_patterns': scores.get('shadow_patterns_count', 0),
                'op_moves': scores.get('op_moves_total', 0),
                'op_moves_effective': scores.get('op_moves_effective', 0),
                'instructions_omitted': scores.get('instructions_omitted', 0),
                'pre_drift_signals': scores.get('pre_drift_signals_count', 0),
            })
            
            print(f"DONE (score: {overall}/10)")
            
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({'name': name, 'error': str(e)})
    
    # Sort by overall drift score (worst first)
    results.sort(key=lambda r: r['overall_score'], reverse=True)
    
    # Write full results
    results_path = os.path.join(output_dir, "batch_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Write summary report
    summary_path = os.path.join(output_dir, "batch_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BATCH DRIFT AUDIT SUMMARY\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total conversations analyzed: {len(results)}\n")
        f.write(f"Total conversations skipped (< {min_messages} msgs): {len(conversations) - len(eligible)}\n")
        f.write(f"Errors: {len(errors)}\n\n")
        
        if results:
            total_msgs = sum(r['message_count'] for r in results)
            total_flags = sum(r['commission_flags'] + r['omission_flags'] for r in results)
            total_corrections = sum(r['corrections_total'] for r in results)
            total_failed = sum(r['corrections_failed'] for r in results)
            total_voids = sum(r['void_events'] for r in results)
            total_op_moves = sum(r['op_moves'] for r in results)
            avg_score = sum(r['overall_score'] for r in results) / len(results)
            
            f.write("AGGREGATE STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total messages processed: {total_msgs}\n")
            f.write(f"  Average drift score: {avg_score:.1f}/10\n")
            f.write(f"  Total drift flags: {total_flags}\n")
            f.write(f"  Total correction events: {total_corrections}\n")
            f.write(f"  Total corrections failed: {total_failed}\n")
            f.write(f"  Correction failure rate: {total_failed/max(total_corrections,1)*100:.1f}%\n")
            f.write(f"  Total void events: {total_voids}\n")
            f.write(f"  Total operator moves: {total_op_moves}\n")
            f.write("\n")
            
            # Score distribution
            f.write("SCORE DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            for threshold in [(1, 2, "Clean"), (3, 4, "Low Drift"), (5, 6, "Moderate"), 
                              (7, 8, "Elevated"), (9, 10, "Severe")]:
                lo, hi, label = threshold
                count = sum(1 for r in results if lo <= r['overall_score'] <= hi)
                pct = count / len(results) * 100
                bar = "#" * int(pct / 2)
                f.write(f"  {label:12s} ({lo}-{hi}): {count:3d} ({pct:5.1f}%) {bar}\n")
            f.write("\n")
            
            # Top 20 worst drift
            f.write("TOP 20 HIGHEST DRIFT CONVERSATIONS\n")
            f.write("-" * 40 + "\n")
            for r in results[:20]:
                f.write(f"  [{r['overall_score']:2d}/10] {r['name'][:50]}\n")
                f.write(f"         msgs:{r['message_count']} flags:{r['commission_flags']+r['omission_flags']}"
                        f" corr_fail:{r['corrections_failed']}/{r['corrections_total']}"
                        f" voids:{r['void_events']} ops:{r['op_moves']}\n")
            f.write("\n")
            
            # Conversations with failed corrections
            failed_corr = [r for r in results if r['corrections_failed'] > 0]
            f.write(f"CONVERSATIONS WITH FAILED CORRECTIONS ({len(failed_corr)})\n")
            f.write("-" * 40 + "\n")
            for r in sorted(failed_corr, key=lambda x: x['corrections_failed'], reverse=True)[:20]:
                f.write(f"  {r['corrections_failed']}/{r['corrections_total']} failed: {r['name'][:50]}\n")
            f.write("\n")
            
            # Conversations with most operator moves
            op_active = [r for r in results if r['op_moves'] > 0]
            f.write(f"MOST OPERATOR INTERVENTION ({len(op_active)} conversations)\n")
            f.write("-" * 40 + "\n")
            for r in sorted(op_active, key=lambda x: x['op_moves'], reverse=True)[:10]:
                eff = r['op_moves_effective']
                tot = r['op_moves']
                f.write(f"  {tot} moves ({eff} effective): {r['name'][:50]}\n")
        
        if errors:
            f.write(f"\nERRORS ({len(errors)})\n")
            f.write("-" * 40 + "\n")
            for e in errors:
                f.write(f"  {e['name'][:50]}: {e['error'][:80]}\n")
    
    print()
    print("=" * 70)
    print(f"BATCH COMPLETE: {len(results)} conversations audited")
    if results:
        avg = sum(r['overall_score'] for r in results) / len(results)
        print(f"Average drift score: {avg:.1f}/10")
        print(f"Highest drift: {results[0]['name'][:50]} ({results[0]['overall_score']}/10)")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch audit Claude conversations")
    parser.add_argument("export_path", help="Path to conversations.json from Claude data export")
    parser.add_argument("--min-messages", type=int, default=10, help="Minimum messages to audit (default: 10)")
    parser.add_argument("--output", default="batch_results", help="Output directory")
    args = parser.parse_args()
    
    run_batch(args.export_path, args.min_messages, args.output)
