"""
Drift Auditor — Chat Parser
=============================
Parses chat transcripts into structured turns.
Handles Claude JSON, plain text with role markers, Claude app copy-paste format.
Extracted from drift_auditor.py monolith — no logic changes.
"""

import json
import re


def parse_chat_log(raw_text: str) -> list[dict]:
    """
    Parse a chat transcript into structured turns.
    Handles multiple formats:
      - Claude.ai export (JSON)
      - Plain text with role markers (Human:/Assistant:)
      - Custom formats with explicit turn markers

    Returns list of dicts: [{"role": "user"|"assistant", "content": "...", "turn": N}]
    """
    # Canonical role mapping
    ROLE_MAP = {
        "user": "user",
        "human": "user",
        "person": "user",
        "customer": "user",
        "assistant": "assistant",
        "claude": "assistant",
        "model": "assistant",
        "ai": "assistant",
        "bot": "assistant",
        "system": "system",
    }

    def normalize_role(raw_role: str) -> str:
        return ROLE_MAP.get(raw_role.lower().strip(), raw_role.lower().strip())
    # Try JSON first (Claude.ai export format)
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            turns = []
            for i, msg in enumerate(data):
                role = normalize_role(msg.get("role", msg.get("sender", "unknown")))
                content = msg.get("content", msg.get("text", ""))
                if isinstance(content, list):
                    # Handle content blocks
                    content = " ".join(
                        block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                turns.append({"role": role, "content": str(content), "turn": i})
            return turns
        elif isinstance(data, dict) and "chat_messages" in data:
            turns = []
            for i, msg in enumerate(data["chat_messages"]):
                role = normalize_role(msg.get("sender", "unknown"))
                content = msg.get("text", "")
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                turns.append({"role": role, "content": str(content), "turn": i})
            return turns
    except (json.JSONDecodeError, TypeError):
        pass

    # Try plain text with role markers
    turns = []
    # Split on common role markers
    pattern = r'(?:^|\n)(Human|User|Assistant|Claude|System)\s*:\s*'
    parts = re.split(pattern, raw_text, flags=re.IGNORECASE)

    if len(parts) > 1:
        # parts[0] is text before first marker (often empty)
        i = 1  # skip preamble
        turn_num = 0
        while i < len(parts) - 1:
            role_raw = parts[i].strip().lower()
            content = parts[i + 1].strip()
            role = normalize_role(role_raw)
            if role_raw == "system":
                role = "system"
            turns.append({"role": role, "content": content, "turn": turn_num})
            turn_num += 1
            i += 2
        return turns

    # Try Claude app copy-paste format
    # Pattern: user message (short), then date line, then optional summary header,
    # then assistant response (longer, often with citation markers like "Wikipedia", "NPR")
    # Date lines look like: "Dec 25, 2025" or "Jan 3, 2026"
    DATE_LINE = r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:,\s+\d{4})?$'

    lines = raw_text.split('\n')

    # Check if this looks like Claude app format (multiple date lines present)
    date_line_count = sum(1 for line in lines if re.match(DATE_LINE, line.strip()))

    if date_line_count >= 2:
        turns = []
        turn_num = 0
        current_role = None
        current_content = []

        # State machine: we identify blocks between date lines
        # The text BEFORE a date line (after the previous block) is typically user input
        # The text AFTER a date line + summary header is model output
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check if this is a date line
            if re.match(DATE_LINE, line):
                # Save whatever we've accumulated as a turn
                if current_content and current_role:
                    content_text = '\n'.join(current_content).strip()
                    if content_text and len(content_text) > 5:
                        turns.append({"role": current_role, "content": content_text, "turn": turn_num})
                        turn_num += 1
                    current_content = []

                # Skip date line
                i += 1

                # Skip blank lines after date
                while i < len(lines) and not lines[i].strip():
                    i += 1

                # The next non-blank line is often a summary header (model's thinking)
                # These are short descriptive lines like "Synthesized atmospheric river conditions"
                # Skip it if it looks like a summary header (short, title-case-ish, no punctuation at end)
                if i < len(lines):
                    header_candidate = lines[i].strip()
                    # Summary headers are typically 3-12 words, no ending punctuation
                    word_count = len(header_candidate.split())
                    if (2 <= word_count <= 15 and
                        not header_candidate.endswith(('.', '?', '!', '"', "'")) and
                        not any(marker in header_candidate.lower() for marker in
                                ['http', 'www', '@', 'i ', "i'", 'you ', 'my ', 'the '])):
                        i += 1  # Skip the summary header

                # Skip blank lines after header
                while i < len(lines) and not lines[i].strip():
                    i += 1

                # Now we're in the assistant response
                current_role = "assistant"
                continue

            # If we're not in a role yet, this is user input
            if current_role is None:
                current_role = "user"

            # Accumulate content
            if line:
                # Check if this line might be a transition to user input
                # User turns in Claude app format tend to be shorter and more informal
                # They come right before a date line
                # Look ahead: if next non-blank line is a date, this block is user
                lookahead = i + 1
                while lookahead < len(lines) and not lines[lookahead].strip():
                    lookahead += 1

                if (lookahead < len(lines) and
                    re.match(DATE_LINE, lines[lookahead].strip()) and
                    current_role == "assistant"):
                    # Save current assistant block
                    content_text = '\n'.join(current_content).strip()
                    if content_text and len(content_text) > 5:
                        turns.append({"role": "assistant", "content": content_text, "turn": turn_num})
                        turn_num += 1
                    current_content = [line]
                    current_role = "user"
                    i += 1
                    continue

                current_content.append(line)

            i += 1

        # Don't forget the last block
        if current_content and current_role:
            content_text = '\n'.join(current_content).strip()
            if content_text and len(content_text) > 5:
                turns.append({"role": current_role, "content": content_text, "turn": turn_num})

        if len(turns) >= 2:
            return turns

    # Fallback: treat entire text as a single block
    return [{"role": "unknown", "content": raw_text, "turn": 0}]
