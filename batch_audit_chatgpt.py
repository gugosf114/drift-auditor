"""
Batch Audit Runner for ChatGPT Data Export.
Converts ChatGPT's nested mapping format to flat messages, then audits.
"""
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from drift_auditor import audit_conversation, format_report, report_to_json


def chatgpt_conv_to_messages(conv: dict) -> list[dict]:
    """
    Convert ChatGPT's nested mapping tree to a flat ordered message list.
    ChatGPT stores messages as a tree (mapping) with parent/children refs.
    We walk the tree from root to current_node to get the linear conversation.
    """
    mapping = conv.get('mapping', {})
    if not mapping:
        return []
    
    # Build parent->children index
    children_map = {}
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get('parent')
        if parent is None:
            root_id = node_id
        else:
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(node_id)
    
    if root_id is None:
        return []
    
    # Walk the tree linearly (follow first child at each level)
    messages = []
    current = root_id
    visited = set()
    
    while current and current not in visited:
        visited.add(current)
        node = mapping.get(current, {})
        msg = node.get('message')
        
        if msg:
            author = msg.get('author', {}).get('role', 'unknown')
            content_parts = msg.get('content', {}).get('parts', [])
            
            # Flatten parts to text
            text_parts = []
            for part in content_parts:
                if isinstance(part, str) and part.strip():
                    text_parts.append(part)
                elif isinstance(part, dict) and part.get('text'):
                    text_parts.append(part['text'])
            
            text = '\n'.join(text_parts).strip()
            
            if text and author in ('user', 'assistant'):
                role = 'user' if author == 'user' else 'assistant'
                messages.append({
                    'role': role,
                    'content': text,
                    'turn': len(messages),
                })
        
        # Follow to next node (first child, or use current_node path)
        kids = children_map.get(current, [])
        if kids:
            current = kids[0]
        else:
            break
    
    return messages


def run_batch_chatgpt(export_path: str, min_messages: int = 10, output_dir: str = "batch_results_chatgpt"):
    """Run drift audit on all ChatGPT conversations meeting minimum threshold."""
    
    print(f"Loading ChatGPT conversations from {export_path}...")
    with open(export_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    print(f"Found {len(conversations)} conversations")
    
    # Convert and filter
    eligible = []
    for conv in conversations:
        msgs = chatgpt_conv_to_messages(conv)
        if len(msgs) >= min_messages:
            eligible.append((conv, msgs))
    
    print(f"{len(eligible)} conversations with >= {min_messages} messages")
    print(f"Skipping {len(conversations) - len(eligible)} short conversations")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    errors = []
    
    for i, (conv, msgs) in enumerate(eligible):
        title = conv.get('title', 'unnamed') or 'unnamed'
        safe_title = title[:60].encode('ascii', 'replace').decode('ascii')
        msg_count = len(msgs)
        
        print(f"[{i+1}/{len(eligible)}] {safe_title} ({msg_count} msgs)...", end=" ", flush=True)
        
        try:
            # Convert messages to JSON string for our parser
            raw_text = json.dumps(msgs)
            
            report = audit_conversation(
                raw_text=raw_text,
                conversation_id=f"gpt_{safe_title[:30]}",
                window_size=50,
                overlap=10,
            )
            
            scores = report.summary_scores
            overall = scores.get('overall_drift_score', 0)
            
            results.append({
                'title': title[:60],
                'model': conv.get('default_model_slug', 'unknown'),
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
            errors.append({'title': safe_title, 'error': str(e)})
    
    # Sort by drift score
    results.sort(key=lambda r: r['overall_score'], reverse=True)
    
    # Write results
    results_path = os.path.join(output_dir, "batch_results_chatgpt.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Write summary
    summary_path = os.path.join(output_dir, "batch_summary_chatgpt.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BATCH DRIFT AUDIT SUMMARY — ChatGPT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total conversations analyzed: {len(results)}\n")
        f.write(f"Total skipped (< {min_messages} msgs): {len(conversations) - len(eligible)}\n")
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
            f.write(f"  Total operator moves: {total_op_moves}\n\n")
            
            # Score distribution
            f.write("SCORE DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            for lo, hi, label in [(1, 2, "Clean"), (3, 4, "Low Drift"), (5, 6, "Moderate"),
                                   (7, 8, "Elevated"), (9, 10, "Severe")]:
                count = sum(1 for r in results if lo <= r['overall_score'] <= hi)
                pct = count / len(results) * 100
                bar = "#" * int(pct / 2)
                f.write(f"  {label:12s} ({lo}-{hi}): {count:3d} ({pct:5.1f}%) {bar}\n")
            f.write("\n")
            
            # Top 20
            f.write("TOP 20 HIGHEST DRIFT CONVERSATIONS\n")
            f.write("-" * 40 + "\n")
            for r in results[:20]:
                safe = r['title'].encode('ascii', 'replace').decode('ascii')
                f.write(f"  [{r['overall_score']:2d}/10] [{r['model']}] {safe}\n")
                f.write(f"         msgs:{r['message_count']} flags:{r['commission_flags']+r['omission_flags']}"
                        f" corr_fail:{r['corrections_failed']}/{r['corrections_total']}"
                        f" voids:{r['void_events']} ops:{r['op_moves']}\n")
            f.write("\n")
            
            # Failed corrections
            failed_corr = [r for r in results if r['corrections_failed'] > 0]
            f.write(f"CONVERSATIONS WITH FAILED CORRECTIONS ({len(failed_corr)})\n")
            f.write("-" * 40 + "\n")
            for r in sorted(failed_corr, key=lambda x: x['corrections_failed'], reverse=True)[:20]:
                safe = r['title'].encode('ascii', 'replace').decode('ascii')
                f.write(f"  {r['corrections_failed']}/{r['corrections_total']} failed: {safe}\n")
            f.write("\n")
            
            # Model distribution
            model_counts = {}
            model_scores = {}
            for r in results:
                m = r['model']
                model_counts[m] = model_counts.get(m, 0) + 1
                if m not in model_scores:
                    model_scores[m] = []
                model_scores[m].append(r['overall_score'])
            
            f.write("MODEL DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            for m, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
                avg = sum(model_scores[m]) / len(model_scores[m])
                f.write(f"  {m}: {count} conversations, avg drift {avg:.1f}/10\n")
        
        if errors:
            f.write(f"\nERRORS ({len(errors)})\n")
            f.write("-" * 40 + "\n")
            for e in errors:
                f.write(f"  {e['title']}: {e['error'][:80]}\n")
    
    # Cross-model comparison
    comparison_path = os.path.join(output_dir, "cross_model_comparison.txt")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("CROSS-MODEL DRIFT COMPARISON: Claude vs ChatGPT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")
        
        # Load Claude results if available
        claude_path = os.path.join("batch_results", "batch_results.json")
        if os.path.exists(claude_path):
            with open(claude_path, 'r') as cf:
                claude_results = json.load(cf)
            
            c_avg = sum(r['overall_score'] for r in claude_results) / max(len(claude_results), 1)
            g_avg = sum(r['overall_score'] for r in results) / max(len(results), 1)
            
            c_msgs = sum(r['message_count'] for r in claude_results)
            g_msgs = sum(r['message_count'] for r in results)
            
            c_flags = sum(r['commission_flags'] + r['omission_flags'] for r in claude_results)
            g_flags = sum(r['commission_flags'] + r['omission_flags'] for r in results)
            
            c_corr = sum(r['corrections_total'] for r in claude_results)
            g_corr = sum(r['corrections_total'] for r in results)
            c_fail = sum(r['corrections_failed'] for r in claude_results)
            g_fail = sum(r['corrections_failed'] for r in results)
            
            c_voids = sum(r['void_events'] for r in claude_results)
            g_voids = sum(r['void_events'] for r in results)
            
            f.write(f"{'Metric':<35s} {'Claude':>12s} {'ChatGPT':>12s}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Conversations analyzed':<35s} {len(claude_results):>12d} {len(results):>12d}\n")
            f.write(f"{'Total messages':<35s} {c_msgs:>12d} {g_msgs:>12d}\n")
            f.write(f"{'Average drift score':<35s} {c_avg:>12.1f} {g_avg:>12.1f}\n")
            f.write(f"{'Total drift flags':<35s} {c_flags:>12d} {g_flags:>12d}\n")
            f.write(f"{'Flags per message':<35s} {c_flags/max(c_msgs,1):>12.3f} {g_flags/max(g_msgs,1):>12.3f}\n")
            f.write(f"{'Correction events':<35s} {c_corr:>12d} {g_corr:>12d}\n")
            f.write(f"{'Corrections failed':<35s} {c_fail:>12d} {g_fail:>12d}\n")
            f.write(f"{'Correction failure rate':<35s} {c_fail/max(c_corr,1)*100:>11.1f}% {g_fail/max(g_corr,1)*100:>11.1f}%\n")
            f.write(f"{'Void events':<35s} {c_voids:>12d} {g_voids:>12d}\n")
            f.write(f"{'Voids per message':<35s} {c_voids/max(c_msgs,1):>12.3f} {g_voids/max(g_msgs,1):>12.3f}\n")
            f.write("\n")
            
            # Score distribution comparison
            f.write("SCORE DISTRIBUTION COMPARISON\n")
            f.write("-" * 60 + "\n")
            for lo, hi, label in [(1, 2, "Clean"), (3, 4, "Low Drift"), (5, 6, "Moderate"),
                                   (7, 8, "Elevated"), (9, 10, "Severe")]:
                c_count = sum(1 for r in claude_results if lo <= r['overall_score'] <= hi)
                g_count = sum(1 for r in results if lo <= r['overall_score'] <= hi)
                c_pct = c_count / max(len(claude_results), 1) * 100
                g_pct = g_count / max(len(results), 1) * 100
                f.write(f"  {label:12s} ({lo}-{hi}):  Claude {c_pct:5.1f}%  |  ChatGPT {g_pct:5.1f}%\n")
        else:
            f.write("Claude batch results not found. Run batch_audit.py first.\n")
    
    print()
    print("=" * 70)
    print(f"BATCH COMPLETE: {len(results)} ChatGPT conversations audited")
    if results:
        avg = sum(r['overall_score'] for r in results) / len(results)
        print(f"Average drift score: {avg:.1f}/10")
        print(f"Highest drift: {results[0]['title'][:50]} ({results[0]['overall_score']}/10)")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Comparison: {comparison_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch audit ChatGPT conversations")
    parser.add_argument("export_path", help="Path to ChatGPT conversations.json")
    parser.add_argument("--min-messages", type=int, default=10, help="Minimum messages (default: 10)")
    parser.add_argument("--output", default="batch_results_chatgpt", help="Output directory")
    args = parser.parse_args()
    
    run_batch_chatgpt(args.export_path, args.min_messages, args.output)
