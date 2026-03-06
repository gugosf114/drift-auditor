"""
Drift Auditor — Chat Parser
=============================
Parses chat transcripts into structured turns.
Handles Claude JSON, ChatGPT export, plain text with role markers,
Claude app copy-paste format.
Extracted from drift_auditor.py monolith.
"""

import json
import re


class ParseError(Exception):
    """Raised when the parser can identify the format but can't extract turns."""
    pass


# Canonical role mapping — shared across all format handlers
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
    "chatgpt": "assistant",
    "gpt": "assistant",
    "system": "system",
}


def _normalize_role(raw_role: str) -> str:
    return ROLE_MAP.get(raw_role.lower().strip(), raw_role.lower().strip())


def _flatten_content(content) -> str:
    """Safely flatten content that might be a string, list of blocks, or dict."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str) and block.strip():
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return " ".join(parts)
    if isinstance(content, dict):
        # ChatGPT content.parts format
        parts = content.get("parts", [])
        return _flatten_content(parts)
    return str(content) if content else ""


def _parse_chatgpt_mapping(data: dict) -> list[dict] | None:
    """
    Parse ChatGPT's nested mapping tree format.
    ChatGPT stores messages as a tree (mapping) with parent/children refs.
    Walk from current_node up to root, reverse for chronological order.
    """
    if isinstance(data, list):
        # ChatGPT full export is a list of conversations
        # If someone uploads the whole export, try the first conversation
        for conv in data:
            if isinstance(conv, dict) and "mapping" in conv:
                result = _parse_chatgpt_mapping(conv)
                if result and len(result) >= 2:
                    return result
        return None

    mapping = data.get("mapping")
    if not mapping or not isinstance(mapping, dict):
        return None

    current_id = data.get("current_node")
    if not current_id or current_id not in mapping:
        current_id = next(
            (nid for nid, n in mapping.items() if n.get("parent") is None), None
        )
    if current_id is None:
        return None

    # Walk parent chain from current_node to root, then reverse
    path = []
    visited = set()
    cur = current_id
    while cur and cur not in visited:
        visited.add(cur)
        node = mapping.get(cur)
        if not node:
            break
        path.append(cur)
        cur = node.get("parent")
    path.reverse()

    turns = []
    for node_id in path:
        node = mapping.get(node_id, {})
        msg = node.get("message")
        if not msg:
            continue
        author = msg.get("author", {}).get("role", "unknown")
        content = _flatten_content(msg.get("content", {}))
        if not content.strip():
            continue
        role = _normalize_role(author)
        if role in ("user", "assistant", "system"):
            turns.append({"role": role, "content": content.strip(), "turn": len(turns)})

    return turns if len(turns) >= 2 else None


def parse_chat_log(raw_text: str) -> list[dict]:
    """
    Parse a chat transcript into structured turns.
    Handles multiple formats:
      - Claude.ai export (JSON list or chat_messages wrapper)
      - ChatGPT data export (nested mapping format)
      - Plain text with role markers (Human:/Assistant:)
      - Claude app copy-paste (date-separated)

    Returns list of dicts: [{"role": "user"|"assistant", "content": "...", "turn": N}]
    Raises ParseError if format is recognized but extraction fails.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        raise ParseError("Empty input — no conversation text provided.")

    # Try JSON formats first
    try:
        data = json.loads(raw_text)

        # ChatGPT mapping format (single conversation or full export)
        if isinstance(data, dict) and "mapping" in data:
            result = _parse_chatgpt_mapping(data)
            if result:
                return result
            raise ParseError(
                "Recognized ChatGPT export format but could not extract messages. "
                "The conversation may be empty or contain only system messages."
            )

        if isinstance(data, list):
            # Could be ChatGPT full export (list of conversations with mapping)
            if data and isinstance(data[0], dict) and "mapping" in data[0]:
                result = _parse_chatgpt_mapping(data)
                if result:
                    return result
                raise ParseError(
                    "Recognized ChatGPT multi-conversation export but no conversations "
                    "had extractable messages. Try uploading a single conversation."
                )

            # Claude.ai JSON export (list of message objects)
            turns = []
            for i, msg in enumerate(data):
                if not isinstance(msg, dict):
                    continue
                role = _normalize_role(msg.get("role", msg.get("sender", "unknown")))
                content = _flatten_content(msg.get("content", msg.get("text", "")))
                if content.strip():
                    turns.append({"role": role, "content": content.strip(), "turn": len(turns)})
            if turns:
                return turns
            raise ParseError(
                "Parsed JSON array but no messages found. Expected objects with "
                "'role'/'content' or 'sender'/'text' fields."
            )

        elif isinstance(data, dict):
            # Claude chat_messages wrapper
            if "chat_messages" in data:
                turns = []
                for msg in data["chat_messages"]:
                    if not isinstance(msg, dict):
                        continue
                    role = _normalize_role(msg.get("sender", "unknown"))
                    content = _flatten_content(msg.get("text", ""))
                    if content.strip():
                        turns.append({"role": role, "content": content.strip(), "turn": len(turns)})
                if turns:
                    return turns
                raise ParseError(
                    "Found 'chat_messages' key but no extractable messages inside."
                )

            # Generic JSON object — try common wrapper keys
            for key in ("messages", "conversation", "turns", "data"):
                if key in data and isinstance(data[key], list):
                    turns = []
                    for msg in data[key]:
                        if not isinstance(msg, dict):
                            continue
                        role = _normalize_role(
                            msg.get("role", msg.get("sender", msg.get("author", "unknown")))
                        )
                        content = _flatten_content(
                            msg.get("content", msg.get("text", msg.get("message", "")))
                        )
                        if content.strip():
                            turns.append({"role": role, "content": content.strip(), "turn": len(turns)})
                    if turns:
                        return turns

            raise ParseError(
                "Parsed JSON object but couldn't find messages. Expected a 'chat_messages', "
                "'messages', 'conversation', or 'mapping' key containing message objects."
            )

    except json.JSONDecodeError:
        pass  # Not JSON — fall through to text parsing
    except ParseError:
        raise  # Re-raise our own errors

    # Try plain text with role markers
    turns = []
    # Split on common role markers (expanded set)
    pattern = r'(?:^|\n)(Human|User|Assistant|Claude|System|AI|Bot|Model|ChatGPT|GPT)\s*:\s*'
    parts = re.split(pattern, raw_text, flags=re.IGNORECASE)

    if len(parts) > 1:
        # parts[0] is text before first marker (often empty)
        i = 1  # skip preamble
        while i < len(parts) - 1:
            role_raw = parts[i].strip().lower()
            content = parts[i + 1].strip()
            role = _normalize_role(role_raw)
            if content:
                turns.append({"role": role, "content": content, "turn": len(turns)})
            i += 2
        if turns:
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

    # Fallback: try line-by-line heuristic for conversations that use
    # ">" quoting, markdown headers, or numbered turns
    bracket_pattern = r'^>\s*(.*?)$'
    bracket_lines = re.findall(bracket_pattern, raw_text, re.MULTILINE)
    if len(bracket_lines) >= 2:
        turns = []
        for line in bracket_lines:
            role = "user" if len(turns) % 2 == 0 else "assistant"
            if line.strip():
                turns.append({"role": role, "content": line.strip(), "turn": len(turns)})
        if len(turns) >= 2:
            return turns

    # Last resort: if there's enough text, return it as unparsed so the
    # caller gets a clear signal rather than silent failure
    if len(raw_text) < 20:
        raise ParseError(
            "Input too short to be a conversation. "
            "Provide a transcript with Human:/Assistant: markers or a JSON export."
        )
    raise ParseError(
        "Could not detect conversation format. Supported formats:\n"
        "- JSON: Claude.ai export, ChatGPT data export, or {role, content} objects\n"
        "- Text: Lines prefixed with Human:/Assistant:, User:/Claude:, etc.\n"
        "- Claude app: Copy-paste with date separators"
    )
