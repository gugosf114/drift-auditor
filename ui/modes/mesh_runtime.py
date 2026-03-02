"""
Drift Auditor — Mesh Runtime Mode
=================================
Build/load relationship mesh index and run state-aware retrieval + control frame generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from relationship_mesh import (
    compile_bootstrap_prompt,
    RelationshipState,
    build_control_frame,
    build_mesh_index,
    compile_episode,
    compile_mesh_from_files,
    list_transcript_files,
    mesh_index_from_payload,
    mesh_index_to_payload,
    query_mesh,
    update_relationship_state,
)


@st.cache_data(show_spinner=False)
def compile_mesh_from_uploads_cached(
    uploads: tuple[tuple[str, bytes], ...],
    include_audit: bool,
    normalize_voice: bool,
) -> dict:
    episodes = []
    for name, data in uploads:
        raw = data.decode("utf-8", errors="ignore")
        episodes.append(
            compile_episode(
                raw,
                source_name=name,
                raw_text_path=name,
                include_audit=include_audit,
                normalize_voice=normalize_voice,
            )
        )
    index = build_mesh_index(episodes)
    return mesh_index_to_payload(index)


@st.cache_data(show_spinner=False)
def compile_mesh_from_paths_cached(
    paths: tuple[str, ...],
    include_audit: bool,
    normalize_voice: bool,
) -> dict:
    index = compile_mesh_from_files(
        list(paths), include_audit=include_audit, normalize_voice=normalize_voice
    )
    return mesh_index_to_payload(index)


def _state_from_config(config: dict) -> RelationshipState:
    return RelationshipState(
        trust_level=float(config.get("mesh_trust", 0.7)),
        urgency_level=float(config.get("mesh_urgency", 0.5)),
        autonomy_preference=float(config.get("mesh_autonomy", 0.7)),
        verbosity_tolerance=float(config.get("mesh_verbosity", 0.3)),
        evidence_strictness=float(config.get("mesh_evidence", 0.8)),
        correction_debt=float(config.get("mesh_debt", 0.0)),
        unresolved_threads=int(config.get("mesh_unresolved", 0)),
        risk_posture=float(config.get("mesh_risk", 0.8)),
        continuity_confidence=float(config.get("mesh_continuity", 0.8)),
    )


def render_mesh_runtime_mode(config: dict) -> None:
    st.markdown("## 🕸️ Mesh Runtime")
    st.caption(
        "Lossless persistence layer: compile episodes, query across nonlinear context, and generate injectable control frames."
    )

    uploads = config.get("mesh_uploaded_files") or []
    mesh_index_upload = config.get("mesh_index_upload")
    include_audit = bool(config.get("mesh_include_audit", True))
    normalize_voice = bool(config.get("mesh_normalize_voice", True))
    query = (config.get("mesh_query") or "").strip()
    top_k = int(config.get("mesh_top_k", 8))

    repo_root = Path(__file__).resolve().parents[2]
    home_dir = Path.home()
    preset_map = {
        "Downloads (.txt/.json)": str(home_dir / "Downloads"),
        "drift-auditor/adversarial_results": str(repo_root / "adversarial_results"),
        "drift-auditor/examples": str(repo_root / "examples"),
    }

    st.markdown("### One-Click Batch Import")
    p1, p2, p3 = st.columns([2, 1, 1])
    with p1:
        preset_name = st.selectbox("Preset source", list(preset_map.keys()), index=0)
    with p2:
        preset_recursive = st.checkbox("Recursive", value=True)
    with p3:
        preset_max_files = st.number_input("Max files", min_value=1, max_value=5000, value=500, step=25)

    manual_folder = st.text_input(
        "Or custom folder path",
        value="",
        placeholder=r"C:\Users\georg\Downloads",
        help="Leave empty to use preset path above.",
    ).strip()

    load_col, clear_col = st.columns([1, 1])
    with load_col:
        load_clicked = st.button("Load Batch Preset", use_container_width=True)
    with clear_col:
        clear_clicked = st.button("Clear Batch Files", use_container_width=True)

    if clear_clicked:
        st.session_state.pop("mesh_preset_paths", None)

    if load_clicked:
        chosen_root = manual_folder if manual_folder else preset_map[preset_name]
        paths = list_transcript_files(
            chosen_root,
            recursive=bool(preset_recursive),
            max_files=int(preset_max_files),
        )
        st.session_state["mesh_preset_paths"] = paths

    preset_paths = st.session_state.get("mesh_preset_paths", [])
    if preset_paths:
        st.info(f"Preset loaded: {len(preset_paths)} file(s)")
        st.text_area(
            "Batch file preview",
            value="\n".join(preset_paths[:20]),
            height=140,
            key="mesh_paths_preview",
        )

    payload: dict | None = None
    source_label = ""

    if mesh_index_upload is not None:
        try:
            payload = json.loads(mesh_index_upload.getvalue().decode("utf-8", errors="ignore"))
            source_label = f"Loaded mesh index JSON: {mesh_index_upload.name}"
        except Exception as exc:
            st.error(f"Unable to parse mesh index JSON: {exc}")
            return
    elif uploads:
        upload_tuple = tuple((f.name, f.getvalue()) for f in uploads)
        payload = compile_mesh_from_uploads_cached(upload_tuple, include_audit, normalize_voice)
        source_label = f"Compiled mesh from {len(uploads)} transcript(s)."
    elif preset_paths:
        payload = compile_mesh_from_paths_cached(
            tuple(preset_paths), include_audit=include_audit, normalize_voice=normalize_voice
        )
        source_label = f"Compiled mesh from preset batch: {len(preset_paths)} transcript(s)."

    if payload is None:
        st.info("Upload transcript files or a saved mesh index JSON from the sidebar.")
        return

    index = mesh_index_from_payload(payload)
    st.success(source_label)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Episodes", len(index.episodes))
    with c2:
        st.metric("Events", len(index.events))
    with c3:
        st.metric("Text Tokens", len(index.text_index))
    with c4:
        st.metric("Graph Edges", sum(len(v) for v in index.episode_edges.values()))

    state = _state_from_config(config)
    if query:
        hits = query_mesh(index, query, state=state, top_k=top_k)

        # State update from top hit events to keep runtime adaptive
        if hits:
            first_ep = index.episodes.get(hits[0].episode_id)
            if first_ep is not None:
                state = update_relationship_state(state, latest_events=first_ep.events[-10:])

        frame = build_control_frame(state=state, hits=hits, query=query)

        st.markdown("### Retrieval Hits")
        if not hits:
            st.warning("No hits. Try broader query terms or compile more transcripts.")
        else:
            for i, hit in enumerate(hits, start=1):
                with st.expander(f"#{i} • {hit.episode_id} • score {hit.score:.2f}", expanded=(i == 1)):
                    st.write("Reasons:", ", ".join(hit.reasons) if hit.reasons else "none")
                    if hit.matching_entities:
                        st.write("Entities:", ", ".join(hit.matching_entities))
                    if hit.open_loops:
                        st.write("Open loops:", hit.open_loops[:3])
                    st.text_area("Excerpt", value=hit.excerpt, height=120, key=f"mesh_ex_{i}")

        st.markdown("### Control Frame (Injectable)")
        frame_json = json.dumps(frame, indent=2, ensure_ascii=False)
        st.code(frame_json, language="json")
        st.download_button(
            "Download Control Frame JSON",
            data=frame_json.encode("utf-8"),
            file_name="control_frame_mesh_v1.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("### Export Mesh Index")
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    st.download_button(
        "Download Mesh Index JSON",
        data=payload_json.encode("utf-8"),
        file_name="relationship_mesh_index.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("### Bootstrap Compiler")
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        max_words = st.slider("Bootstrap max words", 120, 900, 500, step=20)
    with b2:
        max_rules = st.slider("Max rules", 4, 20, 12, step=1)
    with b3:
        compile_bootstrap_clicked = st.button("Compile Bootstrap", use_container_width=True)

    if compile_bootstrap_clicked:
        compiled = compile_bootstrap_prompt(index, max_words=max_words, max_rules=max_rules)
        st.success(f"Compiled bootstrap: {compiled.word_count} words / budget {compiled.max_words}")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Episodes", compiled.metrics.get("episode_count", 0))
        with m2:
            st.metric("Events", compiled.metrics.get("event_count", 0))
        with m3:
            st.metric("Avg Drift", compiled.metrics.get("avg_drift", 0))
        with m4:
            st.metric("Rules Selected", compiled.metrics.get("rules_selected", 0))

        st.markdown("**Compiled Bootstrap Prompt**")
        st.code(compiled.prompt_text, language="markdown")

        st.markdown("**Rule Evidence**")
        st.dataframe(
            [
                {
                    "rule_id": r.rule_id,
                    "support": r.support,
                    "evidence_count": r.evidence_count,
                    "text": r.text,
                    "rationale": r.rationale,
                }
                for r in compiled.rules
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Compiled Bootstrap (.md)",
            data=compiled.prompt_text.encode("utf-8"),
            file_name="bootstrap_compiled_v1.md",
            mime="text/markdown",
            use_container_width=True,
        )
