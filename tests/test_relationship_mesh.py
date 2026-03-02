"""
Tests for relationship mesh persistence layer.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relationship_mesh import (
    RelationshipState,
    build_control_frame,
    build_mesh_index,
    compile_bootstrap_prompt,
    compile_mesh_from_folder,
    compile_episode,
    list_transcript_files,
    load_mesh_index,
    query_mesh,
    save_mesh_index,
    update_relationship_state,
)


def _sample_a() -> str:
    return (
        "User: We need to continue Newton protocol and drift detection.\n"
        "Assistant: Done, I can compile the rules.\n"
        "User: You're wrong, I said no summaries.\n"
        "Assistant: You're right, I apologize. I will fix it.\n"
        "User: Also connect this to AI governance and bot detection?\n"
        "Assistant: Completed."
    )


def _sample_b() -> str:
    return (
        "User: Build bakery pricing guardrails for custom cakes and cookies.\n"
        "Assistant: Done, I added a range and messaging rules.\n"
        "User: continue with Google profile updates and Yelp posting\n"
        "Assistant: completed with next steps"
    )


def test_compile_episode_schema():
    ep = compile_episode(_sample_a(), source_name="a.txt", include_audit=False)
    assert ep.episode_id
    assert ep.turn_count >= 4
    assert len(ep.events) == ep.turn_count
    assert isinstance(ep.entities, list)
    assert isinstance(ep.unresolved_loops, list)


def test_build_index_and_query():
    ep1 = compile_episode(_sample_a(), source_name="a.txt", include_audit=False)
    ep2 = compile_episode(_sample_b(), source_name="b.txt", include_audit=False)
    idx = build_mesh_index([ep1, ep2])

    hits = query_mesh(idx, "newton drift governance", top_k=2)
    assert len(hits) >= 1
    assert hits[0].episode_id == ep1.episode_id
    assert hits[0].score > 0


def test_state_update_and_control_frame():
    ep1 = compile_episode(_sample_a(), source_name="a.txt", include_audit=False)
    idx = build_mesh_index([ep1])

    state = RelationshipState(correction_debt=0.2)
    state = update_relationship_state(state, latest_events=ep1.events)
    hits = query_mesh(idx, "fix correction debt", state=state, top_k=1)
    frame = build_control_frame(state=state, hits=hits, query="fix correction debt")

    assert frame["frame_version"] == "mesh-v1"
    assert frame["mode"] in {"act-now", "repair", "explore", "ask-one-blocker"}
    assert "context_bundle" in frame


def test_save_load_roundtrip():
    ep1 = compile_episode(_sample_a(), source_name="a.txt", include_audit=False)
    ep2 = compile_episode(_sample_b(), source_name="b.txt", include_audit=False)
    idx = build_mesh_index([ep1, ep2])

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mesh.json")
        save_mesh_index(idx, path)
        loaded = load_mesh_index(path)

    assert set(loaded.episodes.keys()) == set(idx.episodes.keys())
    assert len(loaded.events) == len(idx.events)
    hits = query_mesh(loaded, "bakery cookies pricing", top_k=2)
    assert len(hits) >= 1


def test_list_transcript_files_and_compile_folder():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.txt").write_text(_sample_a(), encoding="utf-8")
        (root / "b.json").write_text('[{"role":"user","content":"newton"}]', encoding="utf-8")
        sub = root / "sub"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "c.txt").write_text(_sample_b(), encoding="utf-8")
        (sub / "ignore.md").write_text("ignored", encoding="utf-8")

        files_non_recursive = list_transcript_files(root, recursive=False, max_files=20)
        assert len(files_non_recursive) == 2

        files_recursive = list_transcript_files(root, recursive=True, max_files=20)
        assert len(files_recursive) == 3

        idx = compile_mesh_from_folder(root, recursive=True, max_files=20, include_audit=False)
        assert len(idx.episodes) == 3
        hits = query_mesh(idx, "newton")
        assert len(hits) >= 1


def test_compile_bootstrap_prompt_budget_and_rules():
    ep1 = compile_episode(_sample_a(), source_name="a.txt", include_audit=False)
    ep2 = compile_episode(_sample_b(), source_name="b.txt", include_audit=False)
    idx = build_mesh_index([ep1, ep2])

    compiled = compile_bootstrap_prompt(idx, max_words=220, max_rules=8)
    assert compiled.word_count <= 220
    assert compiled.metrics["episode_count"] == 2
    assert len(compiled.rules) >= 4
    assert "ACTION -> RESULT -> NEXT" in compiled.prompt_text
