"""Tests for ui/theme.py — pure functions, no Streamlit needed."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ui"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from theme import THEMES, DEFAULT_THEME, score_color, score_label, _build_css, get_plotly_layout


class TestScoreColor:
    def test_low_score_is_green(self):
        t = THEMES["Ember"]
        assert score_color(1, t) == t["green"]
        assert score_color(3, t) == t["green"]

    def test_medium_score_is_accent(self):
        t = THEMES["Ember"]
        assert score_color(4, t) == t["accent"]
        assert score_color(5, t) == t["accent"]

    def test_high_score_is_red(self):
        t = THEMES["Ember"]
        assert score_color(6, t) == t["red"]
        assert score_color(7, t) == t["red"]

    def test_severe_score_is_deep_red(self):
        t = THEMES["Ember"]
        assert score_color(8, t) == t["deep_red"]
        assert score_color(10, t) == t["deep_red"]

    def test_defaults_to_ember_theme(self):
        """score_color without a theme arg should not raise."""
        result = score_color(5)
        assert isinstance(result, str)
        assert result.startswith("#")

    def test_all_themes_return_string(self):
        for theme_name, t in THEMES.items():
            for score in [1, 3, 5, 7, 10]:
                result = score_color(score, t)
                assert isinstance(result, str), f"Failed for theme={theme_name}, score={score}"
                assert result.startswith("#")


class TestScoreLabel:
    def test_clean(self):
        assert score_label(1) == "Clean"
        assert score_label(2) == "Clean"

    def test_low_drift(self):
        assert score_label(3) == "Low Drift"
        assert score_label(4) == "Low Drift"

    def test_moderate(self):
        assert score_label(5) == "Moderate"
        assert score_label(6) == "Moderate"

    def test_elevated(self):
        assert score_label(7) == "Elevated"
        assert score_label(8) == "Elevated"

    def test_severe(self):
        assert score_label(9) == "Severe"
        assert score_label(10) == "Severe"


class TestBuildCss:
    def test_returns_non_empty_string(self):
        css = _build_css(THEMES["Ember"])
        assert isinstance(css, str)
        assert len(css) > 100

    def test_contains_style_tags(self):
        css = _build_css(THEMES["Ember"])
        assert "<style>" in css
        assert "</style>" in css

    def test_contains_expected_classes(self):
        css = _build_css(THEMES["Ember"])
        assert ".glass-card" in css
        assert ".metric-value" in css
        assert ".metric-label" in css

    def test_injects_theme_colors(self):
        t = THEMES["Ember"]
        css = _build_css(t)
        assert t["bg"] in css
        assert t["accent"] in css

    def test_all_themes_generate_valid_css(self):
        for theme_name, t in THEMES.items():
            css = _build_css(t)
            assert "<style>" in css, f"Missing <style> for theme={theme_name}"
            assert t["bg"] in css, f"Missing bg color for theme={theme_name}"


class TestGetPlotlyLayout:
    def test_returns_dict(self):
        layout = get_plotly_layout(THEMES["Ember"])
        assert isinstance(layout, dict)

    def test_contains_required_keys(self):
        layout = get_plotly_layout(THEMES["Ember"])
        assert "paper_bgcolor" in layout
        assert "plot_bgcolor" in layout
        assert "font" in layout
        assert "margin" in layout

    def test_font_uses_theme_chart_text(self):
        t = THEMES["Midnight"]
        layout = get_plotly_layout(t)
        assert layout["font"]["color"] == t["chart_text"]


class TestThemesStructure:
    def test_all_themes_have_required_keys(self):
        required = [
            "bg", "surface", "border", "border_accent", "text", "muted",
            "chart_text", "accent", "accent_glow", "accent_hover",
            "green", "red", "deep_red",
        ]
        for theme_name, t in THEMES.items():
            for key in required:
                assert key in t, f"Theme '{theme_name}' missing key '{key}'"

    def test_default_theme_exists(self):
        assert DEFAULT_THEME in THEMES

    def test_five_themes_defined(self):
        assert len(THEMES) == 5
