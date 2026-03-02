"""
Relationship Mesh Persistence Layer
===================================

Lossless conversational persistence for nonlinear operators.

Design goals:
1. Preserve raw transcripts (no compression required)
2. Compile interaction dynamics into reusable state + graph structures
3. Retrieve across domains using relationship signals, not only lexical overlap
4. Generate an injectable control frame per query/session
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from drift_auditor import audit_conversation
from parsers.chat_parser import parse_chat_log


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower()).strip()


def _tokenize(text: str) -> list[str]:
    tokens = [_normalize_token(t) for t in re.split(r"\s+", text)]
    return [t for t in tokens if len(t) >= 2]


_VOICE_NOISE_MAP = {
    "diagnotic": "diagnostic",
    "puppeter": "puppeteer",
    "spasibp": "spasibo",
    "the": "the",
    "wprkflow": "workflow",
    "opetayion": "operation",
}


def normalize_voice_noise(text: str, custom_map: dict[str, str] | None = None) -> str:
    """Normalize common voice-to-text artifacts while preserving original transcript elsewhere."""
    mapping = dict(_VOICE_NOISE_MAP)
    if custom_map:
        mapping.update(custom_map)

    normalized = text
    for src, dst in mapping.items():
        # word-boundary replacement keeps punctuation, newlines, and role markers intact
        normalized = re.sub(rf"\b{re.escape(src)}\b", dst, normalized, flags=re.IGNORECASE)
    return normalized


@dataclass
class RelationshipState:
    """Live dyad state used to shape retrieval and control behavior."""

    trust_level: float = 0.7
    urgency_level: float = 0.5
    autonomy_preference: float = 0.7
    verbosity_tolerance: float = 0.3
    evidence_strictness: float = 0.8
    correction_debt: float = 0.0
    unresolved_threads: int = 0
    risk_posture: float = 0.8
    continuity_confidence: float = 0.8
    updated_at: str = field(default_factory=_utc_now_iso)


@dataclass
class MeshEvent:
    event_id: str
    episode_id: str
    turn: int
    role: str
    event_type: str
    text: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeshEpisode:
    episode_id: str
    source_name: str
    created_at: str
    raw_text_hash: str
    raw_text_path: str | None
    turn_count: int
    turns: list[dict[str, Any]] = field(default_factory=list)
    events: list[MeshEvent] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    shorthand_terms: list[str] = field(default_factory=list)
    unresolved_loops: list[str] = field(default_factory=list)
    audit_scores: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    episode_id: str
    score: float
    reasons: list[str] = field(default_factory=list)
    matching_entities: list[str] = field(default_factory=list)
    open_loops: list[str] = field(default_factory=list)
    excerpt: str = ""


@dataclass
class BootstrapRule:
    rule_id: str
    text: str
    support: float
    evidence_count: int
    rationale: str


@dataclass
class BootstrapCompilation:
    prompt_text: str
    word_count: int
    max_words: int
    rules: list[BootstrapRule] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipMeshIndex:
    episodes: dict[str, MeshEpisode] = field(default_factory=dict)
    events: dict[str, MeshEvent] = field(default_factory=dict)
    text_index: dict[str, set[str]] = field(default_factory=dict)  # token -> event_ids
    entity_index: dict[str, set[str]] = field(default_factory=dict)  # entity -> episode_ids
    tag_index: dict[str, set[str]] = field(default_factory=dict)  # tag -> episode_ids
    open_loop_index: dict[str, set[str]] = field(default_factory=dict)  # token -> episode_ids
    episode_edges: dict[str, dict[str, float]] = field(default_factory=dict)  # weighted adjacency
    built_at: str = field(default_factory=_utc_now_iso)


def _event_type_for_turn(role: str, content: str) -> str:
    lower = content.lower()
    if role == "user":
        if any(phrase in lower for phrase in ["you're wrong", "you are wrong", "no,", "i said", "correct"]):
            return "correction"
        if any(phrase in lower for phrase in ["continue", "pick up", "resume", "as we were"]):
            return "resume"
        if any(phrase in lower for phrase in ["forget it", "drop it", "move on", "kill it"]):
            return "abandon"
        if any(phrase in lower for phrase in ["do ", "run ", "build ", "fix ", "create "]):
            return "directive"
        if "?" in content:
            return "query"
        return "context"
    if role == "assistant":
        if any(phrase in lower for phrase in ["you're right", "you are right", "i was wrong", "apolog"]):
            return "repair"
        if any(phrase in lower for phrase in ["done", "completed", "fixed", "shipped", "implemented"]):
            return "success"
        return "response"
    return "other"


def _extract_entities(text: str) -> list[str]:
    """Lightweight entity extraction tuned for tooling/projects/domains."""
    entities: set[str] = set()
    # Keep product/tool tokens and known proper-case compounds
    tool_matches = re.findall(
        r"\b(?:Claude|ChatGPT|Gemini|Grok|NotebookLM|Windows-MCP|Playwright|Puppeteer|Stripe|Firestore|Pub/Sub|Google|Yelp|Pinterest|Instagram|LinkedIn|Drift Auditor|Audit Ledger|Bakers Agent|Newton|Agora)\b",
        text,
        flags=re.IGNORECASE,
    )
    for m in tool_matches:
        entities.add(m.lower())

    # Domain-like strings
    for m in re.findall(r"\b[a-z0-9-]+\.[a-z]{2,}\b", text.lower()):
        entities.add(m)

    return sorted(entities)


def _extract_unresolved_loops(turns: list[dict[str, Any]]) -> list[str]:
    """Collect unresolved prompts/questions likely to need reopening."""
    loops: list[str] = []
    for turn in turns:
        if turn.get("role") != "user":
            continue
        content = str(turn.get("content", "")).strip()
        lower = content.lower()
        if (
            content.endswith("?")
            or "need to" in lower
            or "unfinished" in lower
            or "pending" in lower
            or "open loop" in lower
            or "next" in lower
        ):
            loops.append(content[:300])
    return loops


def _extract_shorthand_terms(text: str) -> list[str]:
    candidates = [
        "newton",
        "agora",
        "checkpoint",
        "bootloader",
        "operator load",
        "drift",
        "mesh",
        "tiger tamer",
        "kill switch",
    ]
    lower = text.lower()
    return sorted({c for c in candidates if c in lower})


def compile_episode(
    raw_text: str,
    *,
    source_name: str,
    episode_id: str | None = None,
    raw_text_path: str | None = None,
    include_audit: bool = True,
    normalize_voice: bool = True,
) -> MeshEpisode:
    """Compile one transcript into a mesh episode (schema + events + optional audit)."""
    prepared = normalize_voice_noise(raw_text) if normalize_voice else raw_text
    turns = parse_chat_log(prepared)
    eid = episode_id or _sha256_text(source_name + raw_text)[:16]

    events: list[MeshEvent] = []
    for turn in turns:
        turn_no = int(turn.get("turn", 0))
        role = str(turn.get("role", "unknown"))
        content = str(turn.get("content", ""))
        event_type = _event_type_for_turn(role, content)

        event = MeshEvent(
            event_id=f"{eid}:{turn_no}",
            episode_id=eid,
            turn=turn_no,
            role=role,
            event_type=event_type,
            text=content,
            tags=[event_type],
            metadata={"length": len(content)},
        )
        events.append(event)

    audit_scores: dict[str, Any] = {}
    if include_audit:
        try:
            report = audit_conversation(prepared, conversation_id=eid)
            audit_scores = dict(report.summary_scores)
            # Add known tags into event-level metadata for richer retrieval
            tag_events = [
                *report.commission_flags,
                *report.omission_flags,
                *report.pre_drift_signals,
            ]
            for flag in tag_events:
                if flag.turn < len(events):
                    events[flag.turn].tags.append(str(flag.tag or flag.layer or "drift"))
        except Exception:
            # Keep episode compile robust even if audit fails on malformed input.
            audit_scores = {"compile_warning": "audit_failed"}

    entities = _extract_entities(prepared)
    shorthand = _extract_shorthand_terms(prepared)
    unresolved_loops = _extract_unresolved_loops(turns)

    return MeshEpisode(
        episode_id=eid,
        source_name=source_name,
        created_at=_utc_now_iso(),
        raw_text_hash=_sha256_text(raw_text),
        raw_text_path=raw_text_path,
        turn_count=len(turns),
        turns=turns,
        events=events,
        entities=entities,
        shorthand_terms=shorthand,
        unresolved_loops=unresolved_loops,
        audit_scores=audit_scores,
    )


def _index_add(mapping: dict[str, set[str]], key: str, value: str) -> None:
    if not key:
        return
    mapping.setdefault(key, set()).add(value)


def build_mesh_index(episodes: list[MeshEpisode]) -> RelationshipMeshIndex:
    """Build searchable/graph index for nonlinear retrieval."""
    idx = RelationshipMeshIndex()

    for ep in episodes:
        idx.episodes[ep.episode_id] = ep

        for ev in ep.events:
            idx.events[ev.event_id] = ev

            for tok in _tokenize(ev.text):
                _index_add(idx.text_index, tok, ev.event_id)

            for tag in ev.tags:
                _index_add(idx.tag_index, str(tag).lower(), ep.episode_id)

        for ent in ep.entities:
            _index_add(idx.entity_index, ent.lower(), ep.episode_id)

        for loop in ep.unresolved_loops:
            for tok in _tokenize(loop):
                _index_add(idx.open_loop_index, tok, ep.episode_id)

    # Build weighted episode graph by entity/tag overlap
    ep_ids = list(idx.episodes.keys())
    for i, left_id in enumerate(ep_ids):
        left = idx.episodes[left_id]
        left_entities = set(e.lower() for e in left.entities)
        left_tags = set(t for ev in left.events for t in [tag.lower() for tag in ev.tags])

        for right_id in ep_ids[i + 1 :]:
            right = idx.episodes[right_id]
            right_entities = set(e.lower() for e in right.entities)
            right_tags = set(t for ev in right.events for t in [tag.lower() for tag in ev.tags])

            entity_overlap = len(left_entities & right_entities)
            tag_overlap = len(left_tags & right_tags)
            weight = (entity_overlap * 1.5) + tag_overlap

            if weight > 0:
                idx.episode_edges.setdefault(left_id, {})[right_id] = float(weight)
                idx.episode_edges.setdefault(right_id, {})[left_id] = float(weight)

    return idx


def _freshness_boost(episode: MeshEpisode, max_turns: int) -> float:
    if max_turns <= 0:
        return 0.0
    return min(1.0, episode.turn_count / max_turns)


def query_mesh(
    index: RelationshipMeshIndex,
    query: str,
    *,
    state: RelationshipState | None = None,
    top_k: int = 8,
) -> list[RetrievalHit]:
    """State-aware retrieval across the relationship mesh."""
    if not index.episodes:
        return []

    state = state or RelationshipState()
    query_tokens = _tokenize(normalize_voice_noise(query))
    if not query_tokens:
        return []

    # Candidate episodes from any index hit
    candidate_eps: set[str] = set()
    matched_events: dict[str, set[str]] = {}

    for tok in query_tokens:
        for event_id in index.text_index.get(tok, set()):
            ev = index.events[event_id]
            candidate_eps.add(ev.episode_id)
            matched_events.setdefault(ev.episode_id, set()).add(event_id)
        candidate_eps |= index.entity_index.get(tok, set())
        candidate_eps |= index.open_loop_index.get(tok, set())
        candidate_eps |= index.tag_index.get(tok, set())

    if not candidate_eps:
        candidate_eps = set(index.episodes.keys())

    max_turns = max((ep.turn_count for ep in index.episodes.values()), default=1)
    hits: list[RetrievalHit] = []

    for eid in candidate_eps:
        ep = index.episodes[eid]
        reasons: list[str] = []
        score = 0.0

        # Lexical coverage
        event_ids = matched_events.get(eid, set())
        lexical = len(event_ids)
        if lexical:
            score += lexical * 1.4
            reasons.append(f"lexical:{lexical}")

        # Entity matching
        ep_entities = {e.lower() for e in ep.entities}
        matched_entities = sorted(ep_entities & set(query_tokens))
        if matched_entities:
            score += len(matched_entities) * 2.0
            reasons.append(f"entities:{len(matched_entities)}")

        # Open loops relevance
        loop_hits = 0
        for loop in ep.unresolved_loops:
            loop_tokens = set(_tokenize(loop))
            if loop_tokens & set(query_tokens):
                loop_hits += 1
        if loop_hits:
            score += loop_hits * 1.8
            reasons.append(f"open_loops:{loop_hits}")

        # Relationship-state weighting
        if state.correction_debt > 0.4:
            correction_events = sum(1 for ev in ep.events if ev.event_type == "correction")
            if correction_events:
                bonus = correction_events * (1.0 + state.correction_debt)
                score += bonus
                reasons.append(f"correction_debt_bonus:{bonus:.2f}")

        if state.urgency_level > 0.6:
            fresh = _freshness_boost(ep, max_turns)
            score += fresh * 1.5
            reasons.append(f"urgency_freshness:{fresh:.2f}")

        if state.evidence_strictness > 0.7:
            compliance_terms = {"evidence", "audit", "legal", "compliance", "citation"}
            compliance_match = len(compliance_terms & set(_tokenize(" ".join(e.text for e in ep.events[:8]))))
            if compliance_match:
                score += compliance_match * 0.8
                reasons.append(f"evidence_alignment:{compliance_match}")

        # Graph neighborhood boost for nonlinear cross-domain jumps
        neighborhood = index.episode_edges.get(eid, {})
        graph_bonus = sum(neighborhood.values()) * 0.1
        if graph_bonus:
            score += graph_bonus
            reasons.append(f"graph:{graph_bonus:.2f}")

        # Build excerpt from top matching events
        excerpt_parts: list[str] = []
        for event_id in sorted(event_ids)[:3]:
            excerpt_parts.append(index.events[event_id].text[:220])
        if not excerpt_parts and ep.events:
            excerpt_parts = [ep.events[0].text[:220]]

        hits.append(
            RetrievalHit(
                episode_id=eid,
                score=score,
                reasons=reasons,
                matching_entities=matched_entities,
                open_loops=ep.unresolved_loops[:5],
                excerpt="\n---\n".join(excerpt_parts),
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def update_relationship_state(
    state: RelationshipState,
    *,
    latest_events: list[MeshEvent],
) -> RelationshipState:
    """Update live state from recent interaction signals."""
    if not latest_events:
        state.updated_at = _utc_now_iso()
        return state

    corrections = sum(1 for e in latest_events if e.event_type == "correction")
    repairs = sum(1 for e in latest_events if e.event_type == "repair")
    directives = sum(1 for e in latest_events if e.event_type == "directive")
    abandons = sum(1 for e in latest_events if e.event_type == "abandon")
    successes = sum(1 for e in latest_events if e.event_type == "success")

    state.correction_debt = max(0.0, state.correction_debt + (corrections * 0.08) - (repairs * 0.1) - (successes * 0.03))
    state.trust_level = min(1.0, max(0.0, state.trust_level + (successes * 0.03) + (repairs * 0.02) - (abandons * 0.05)))
    state.urgency_level = min(1.0, max(0.0, state.urgency_level + (directives * 0.02) + (corrections * 0.01) - 0.01))
    state.unresolved_threads = max(0, state.unresolved_threads + corrections - repairs - successes)
    state.updated_at = _utc_now_iso()
    return state


def build_control_frame(
    *,
    state: RelationshipState,
    hits: list[RetrievalHit],
    query: str,
    max_excerpts: int = 4,
) -> dict[str, Any]:
    """
    Build injectable per-turn frame.

    This is the runtime payload to prepend/attach for generation control.
    """
    mode = "act-now"
    if state.correction_debt > 0.5:
        mode = "repair"
    elif state.urgency_level < 0.3:
        mode = "explore"
    elif state.trust_level < 0.4:
        mode = "ask-one-blocker"

    context_bundle = []
    for hit in hits[:max_excerpts]:
        context_bundle.append(
            {
                "episode_id": hit.episode_id,
                "score": round(hit.score, 3),
                "reasons": hit.reasons,
                "excerpt": hit.excerpt,
            }
        )

    return {
        "frame_version": "mesh-v1",
        "query": query,
        "mode": mode,
        "state": asdict(state),
        "context_bundle": context_bundle,
        "constraints": {
            "ask_blocking_questions_max": 1,
            "prefer_smallest_reversible_step": True,
            "deterministic_when_high_risk": state.risk_posture > 0.75,
        },
    }


def save_mesh_index(index: RelationshipMeshIndex, path: str | Path) -> None:
    """Persist mesh index as JSON with set-safe encoding."""
    p = Path(path)
    payload = mesh_index_to_payload(index)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_mesh_index(path: str | Path) -> RelationshipMeshIndex:
    """Load persisted index and reconstruct dataclasses."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return mesh_index_from_payload(data)


def mesh_index_to_payload(index: RelationshipMeshIndex) -> dict[str, Any]:
    """Convert mesh index to JSON-safe payload."""
    return {
        "built_at": index.built_at,
        "episodes": {k: asdict(v) for k, v in index.episodes.items()},
        "events": {k: asdict(v) for k, v in index.events.items()},
        "text_index": {k: sorted(list(v)) for k, v in index.text_index.items()},
        "entity_index": {k: sorted(list(v)) for k, v in index.entity_index.items()},
        "tag_index": {k: sorted(list(v)) for k, v in index.tag_index.items()},
        "open_loop_index": {k: sorted(list(v)) for k, v in index.open_loop_index.items()},
        "episode_edges": index.episode_edges,
    }


def mesh_index_from_payload(data: dict[str, Any]) -> RelationshipMeshIndex:
    """Rebuild mesh index from JSON payload dictionary."""

    episodes = {
        k: MeshEpisode(
            **{
                **v,
                "events": [MeshEvent(**ev) for ev in v.get("events", [])],
            }
        )
        for k, v in data.get("episodes", {}).items()
    }
    events = {k: MeshEvent(**v) for k, v in data.get("events", {}).items()}

    return RelationshipMeshIndex(
        episodes=episodes,
        events=events,
        text_index={k: set(v) for k, v in data.get("text_index", {}).items()},
        entity_index={k: set(v) for k, v in data.get("entity_index", {}).items()},
        tag_index={k: set(v) for k, v in data.get("tag_index", {}).items()},
        open_loop_index={k: set(v) for k, v in data.get("open_loop_index", {}).items()},
        episode_edges=data.get("episode_edges", {}),
        built_at=data.get("built_at", _utc_now_iso()),
    )


def compile_mesh_from_files(
    file_paths: list[str | Path],
    *,
    include_audit: bool = True,
    normalize_voice: bool = True,
) -> RelationshipMeshIndex:
    """Convenience helper: compile many transcript files into one mesh index."""
    episodes: list[MeshEpisode] = []
    for fp in file_paths:
        p = Path(fp)
        raw = p.read_text(encoding="utf-8", errors="ignore")
        episodes.append(
            compile_episode(
                raw,
                source_name=p.name,
                raw_text_path=str(p),
                include_audit=include_audit,
                normalize_voice=normalize_voice,
            )
        )
    return build_mesh_index(episodes)


def list_transcript_files(
    root_dir: str | Path,
    *,
    recursive: bool = True,
    max_files: int = 500,
    extensions: tuple[str, ...] = (".txt", ".json"),
) -> list[str]:
    """Discover transcript files under a folder for batch mesh ingest."""
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        return []

    files: list[Path] = []
    if recursive:
        for ext in extensions:
            files.extend(root.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            files.extend(root.glob(f"*{ext}"))

    # deterministic order for reproducibility
    files = sorted({f.resolve() for f in files if f.is_file()}, key=lambda x: str(x).lower())
    return [str(f) for f in files[:max_files]]


def compile_mesh_from_folder(
    root_dir: str | Path,
    *,
    recursive: bool = True,
    max_files: int = 500,
    include_audit: bool = True,
    normalize_voice: bool = True,
) -> RelationshipMeshIndex:
    """Discover transcript files in a folder and compile a mesh index."""
    files = list_transcript_files(root_dir, recursive=recursive, max_files=max_files)
    return compile_mesh_from_files(files, include_audit=include_audit, normalize_voice=normalize_voice)


def _conversation_signals(index: RelationshipMeshIndex) -> dict[str, Any]:
    episodes = list(index.episodes.values())
    events = list(index.events.values())

    directives = sum(1 for e in events if e.event_type == "directive")
    corrections = sum(1 for e in events if e.event_type == "correction")
    repairs = sum(1 for e in events if e.event_type == "repair")
    abandons = sum(1 for e in events if e.event_type == "abandon")
    successes = sum(1 for e in events if e.event_type == "success")
    queries = sum(1 for e in events if e.event_type == "query")

    unresolved = sum(len(ep.unresolved_loops) for ep in episodes)
    avg_unresolved = unresolved / max(1, len(episodes))

    all_text = " ".join(e.text for e in events).lower()
    voice_noise_present = any(token in all_text for token in ["voice", "typo", "dictation", "transcrib", "stutter"])
    high_stakes_present = any(token in all_text for token in ["security", "policy", "admin", "compliance", "legal", "risk"])

    drift_scores = [
        float(ep.audit_scores.get("overall_drift_score", 0))
        for ep in episodes
        if isinstance(ep.audit_scores, dict) and "overall_drift_score" in ep.audit_scores
    ]
    avg_drift = (sum(drift_scores) / len(drift_scores)) if drift_scores else 0.0

    return {
        "episode_count": len(episodes),
        "event_count": len(events),
        "directives": directives,
        "corrections": corrections,
        "repairs": repairs,
        "abandons": abandons,
        "successes": successes,
        "queries": queries,
        "unresolved_total": unresolved,
        "avg_unresolved": avg_unresolved,
        "voice_noise_present": voice_noise_present,
        "high_stakes_present": high_stakes_present,
        "avg_drift": avg_drift,
    }


def _candidate_rules_from_signals(signals: dict[str, Any]) -> list[BootstrapRule]:
    rules: list[BootstrapRule] = []
    total = max(1, int(signals["event_count"]))

    def score(count: int, bonus: float = 0.0) -> float:
        return round(min(1.0, (count / total) * 8 + bonus), 3)

    rules.append(
        BootstrapRule(
            rule_id="R_ACT_FIRST",
            text="Act first when intent is clear; report results after action.",
            support=score(int(signals["directives"]), 0.15),
            evidence_count=int(signals["directives"]),
            rationale="High directive density indicates action-first operator preference.",
        )
    )

    rules.append(
        BootstrapRule(
            rule_id="R_ONE_BLOCKER",
            text="Ask at most one blocking question only when execution is impossible without missing information.",
            support=score(int(signals["queries"]), 0.05),
            evidence_count=int(signals["queries"]),
            rationale="Frequent queries plus correction loops favor constrained clarification behavior.",
        )
    )

    rules.append(
        BootstrapRule(
            rule_id="R_CORRECTION_FAST",
            text="When corrected, acknowledge directly, update course immediately, and avoid defensive explanations.",
            support=score(int(signals["corrections"]), 0.1),
            evidence_count=int(signals["corrections"]),
            rationale="Correction frequency is a strong dyad signal for rapid repair loops.",
        )
    )

    rules.append(
        BootstrapRule(
            rule_id="R_REPAIR_PERSIST",
            text="Repairs must persist across turns; do not repeat resolved errors.",
            support=score(int(signals["repairs"]), 0.08),
            evidence_count=int(signals["repairs"]),
            rationale="Repair events indicate need for persistence guarantees.",
        )
    )

    rules.append(
        BootstrapRule(
            rule_id="R_SMALLEST_REVERSIBLE",
            text="Prefer the smallest reversible step before broader changes.",
            support=score(int(signals["successes"]), 0.12),
            evidence_count=int(signals["successes"]),
            rationale="Success events correlate with incremental execution.",
        )
    )

    rules.append(
        BootstrapRule(
            rule_id="R_CLOSE_OPEN_LOOPS",
            text="Track open loops explicitly and close them before introducing new branches where possible.",
            support=round(min(1.0, signals["avg_unresolved"] / 8), 3),
            evidence_count=int(signals["unresolved_total"]),
            rationale="Unresolved-thread load requires explicit loop management.",
        )
    )

    if signals.get("voice_noise_present"):
        rules.append(
            BootstrapRule(
                rule_id="R_VOICE_NORMALIZE",
                text="Interpret noisy voice-to-text input by intent and context; do not refuse solely due to typos.",
                support=0.92,
                evidence_count=1,
                rationale="Corpus contains repeated voice/transcription artifacts.",
            )
        )

    if signals.get("high_stakes_present"):
        rules.append(
            BootstrapRule(
                rule_id="R_HARD_BOUNDARY",
                text="Respect policy and security boundaries; when blocked, provide safe workaround and explicit handoff steps.",
                support=0.95,
                evidence_count=1,
                rationale="High-stakes operations appear frequently in corpus.",
            )
        )

    return sorted(rules, key=lambda r: r.support, reverse=True)


def compile_bootstrap_prompt(
    index: RelationshipMeshIndex,
    *,
    max_words: int = 500,
    max_rules: int = 12,
    title: str = "SESSION BOOTSTRAP — Auto-Compiled",
) -> BootstrapCompilation:
    """
    Distill relationship mesh into a compact operating bootstrap prompt.

    This is deterministic and evidence-derived (no LLM dependency).
    """
    signals = _conversation_signals(index)
    candidates = _candidate_rules_from_signals(signals)
    chosen = candidates[:max_rules]

    lines: list[str] = [
        title,
        "Mode: high-bandwidth execution with policy-safe boundaries.",
        "",
        "Operator Contract:",
    ]
    for i, rule in enumerate(chosen, start=1):
        lines.append(f"{i}. {rule.text}")

    lines.extend(
        [
            "",
            "Response Format:",
            "ACTION -> RESULT -> NEXT",
            "If blocked: BLOCKER -> REASON -> SAFE WORKAROUND",
        ]
    )

    text = "\n".join(lines).strip()
    words = len(text.split())

    if words > max_words:
        # Trim lowest-support rules until budget fits
        trimmed = chosen[:]
        while words > max_words and len(trimmed) > 4:
            trimmed.pop()
            lines = [
                title,
                "Mode: high-bandwidth execution with policy-safe boundaries.",
                "",
                "Operator Contract:",
            ]
            for i, rule in enumerate(trimmed, start=1):
                lines.append(f"{i}. {rule.text}")
            lines.extend(
                [
                    "",
                    "Response Format:",
                    "ACTION -> RESULT -> NEXT",
                    "If blocked: BLOCKER -> REASON -> SAFE WORKAROUND",
                ]
            )
            text = "\n".join(lines).strip()
            words = len(text.split())
        chosen = trimmed

    metrics = {
        "episode_count": signals["episode_count"],
        "event_count": signals["event_count"],
        "avg_drift": round(float(signals["avg_drift"]), 3),
        "unresolved_total": signals["unresolved_total"],
        "rules_selected": len(chosen),
    }

    return BootstrapCompilation(
        prompt_text=text,
        word_count=words,
        max_words=max_words,
        rules=chosen,
        metrics=metrics,
    )
