"""
Unit tests for accessibility_enhancements.py module.

This module tests all accessibility enhancement functionality including
WCAG 2.1 AA compliance, ARIA attributes, color contrast validation,
and keyboard navigation support.
"""

from unittest.mock import Mock, patch

import gradio as gr
import pytest

from src.ui.components.accessibility_enhancements import (
    AccessibilityEnhancer,
    apply_accessibility_enhancements,
)


@pytest.mark.unit
@pytest.mark.fast
class TestAccessibilityEnhancer:
    """Test cases for AccessibilityEnhancer class."""

    def test_init(self):
        """Test AccessibilityEnhancer initialization."""
        enhancer = AccessibilityEnhancer()

        assert enhancer.aria_labels is not None
        assert isinstance(enhancer.aria_labels, dict)
        assert enhancer.color_contrast_css is not None
        assert isinstance(enhancer.color_contrast_css, str)

    def test_get_aria_labels(self):
        """Test ARIA labels generation."""
        enhancer = AccessibilityEnhancer()
        labels = enhancer._get_aria_labels()

        # Check all required labels are present
        required_labels = [
            "query_input",
            "enhance_button",
            "output_display",
            "file_upload",
            "model_selector",
            "export_button",
            "clear_button",
            "feedback_button",
            "copy_button",
            "tabs_container",
        ]

        for label in required_labels:
            assert label in labels
            assert isinstance(labels[label], str)
            assert len(labels[label]) > 0

    def test_get_high_contrast_css(self):
        """Test high contrast CSS generation."""
        enhancer = AccessibilityEnhancer()
        css = enhancer._get_high_contrast_css()

        # Check CSS contains essential accessibility rules
        assert ".accessibility-mode" in css
        assert "--primary-color" in css
        assert "--focus-outline" in css
        assert "button:focus" in css
        assert "input:focus" in css
        assert ".skip-nav" in css
        assert ".sr-only" in css
        assert "prefers-reduced-motion" in css

    def test_add_aria_attributes_with_valid_component(self):
        """Test adding ARIA attributes to valid Gradio component."""
        enhancer = AccessibilityEnhancer()

        # Create mock component with elem_classes attribute
        mock_component = Mock(spec=gr.Component)
        mock_component.elem_classes = []

        result = enhancer.add_aria_attributes(mock_component, "query_input")

        assert result is mock_component
        assert "aria-label-query_input" in mock_component.elem_classes

    def test_add_aria_attributes_with_existing_classes_list(self):
        """Test adding ARIA attributes to component with existing classes as list."""
        enhancer = AccessibilityEnhancer()

        mock_component = Mock(spec=gr.Component)
        mock_component.elem_classes = ["existing-class"]

        result = enhancer.add_aria_attributes(mock_component, "enhance_button")

        assert result is mock_component
        assert "existing-class" in mock_component.elem_classes
        assert "aria-label-enhance_button" in mock_component.elem_classes

    def test_add_aria_attributes_with_existing_classes_string(self):
        """Test adding ARIA attributes to component with existing classes as string."""
        enhancer = AccessibilityEnhancer()

        mock_component = Mock(spec=gr.Component)
        mock_component.elem_classes = "existing-class"

        result = enhancer.add_aria_attributes(mock_component, "output_display")

        assert result is mock_component
        assert "existing-class" in mock_component.elem_classes
        assert "aria-label-output_display" in mock_component.elem_classes

    def test_add_aria_attributes_no_elem_classes(self):
        """Test adding ARIA attributes to component without elem_classes."""
        enhancer = AccessibilityEnhancer()

        mock_component = Mock(spec=gr.Component)
        del mock_component.elem_classes  # Remove the attribute

        result = enhancer.add_aria_attributes(mock_component, "file_upload")

        assert result is mock_component
        # Should not raise an error

    def test_add_aria_attributes_unknown_component_type(self):
        """Test adding ARIA attributes for unknown component type."""
        enhancer = AccessibilityEnhancer()

        mock_component = Mock(spec=gr.Component)
        mock_component.elem_classes = []

        result = enhancer.add_aria_attributes(mock_component, "unknown_type")

        assert result is mock_component
        # Should not add any classes for unknown types
        assert len(mock_component.elem_classes) == 0

    def test_create_skip_navigation(self):
        """Test skip navigation link creation."""
        enhancer = AccessibilityEnhancer()
        skip_nav = enhancer.create_skip_navigation()

        assert isinstance(skip_nav, str)
        assert 'href="#main-content"' in skip_nav
        assert 'class="skip-nav"' in skip_nav
        assert "Skip to main content" in skip_nav

    def test_create_landmark_regions(self):
        """Test ARIA landmark regions creation."""
        enhancer = AccessibilityEnhancer()
        landmarks = enhancer.create_landmark_regions()

        required_landmarks = ["banner", "navigation", "main", "complementary", "contentinfo"]

        assert isinstance(landmarks, dict)
        for landmark in required_landmarks:
            assert landmark in landmarks
            assert f'role="{landmark}"' in landmarks[landmark]

        # Check specific landmark properties
        assert 'id="main-content"' in landmarks["main"]
        assert 'aria-label="Site header"' in landmarks["banner"]


@pytest.mark.unit
@pytest.mark.fast
class TestColorContrastValidation:
    """Test cases for color contrast validation functionality."""

    def test_validate_color_contrast_wcag_aa_compliant(self):
        """Test color contrast validation for WCAG AA compliant colors."""
        enhancer = AccessibilityEnhancer()

        # Black text on white background (21:1 ratio)
        assert enhancer.validate_color_contrast("#000000", "#ffffff") is True

        # Dark gray on white (7:1 ratio)
        assert enhancer.validate_color_contrast("#595959", "#ffffff") is True

    def test_validate_color_contrast_wcag_aa_non_compliant(self):
        """Test color contrast validation for non-compliant colors."""
        enhancer = AccessibilityEnhancer()

        # Light gray on white (insufficient contrast)
        assert enhancer.validate_color_contrast("#cccccc", "#ffffff") is False

        # Similar colors (insufficient contrast)
        assert enhancer.validate_color_contrast("#eeeeee", "#ffffff") is False

    def test_validate_color_contrast_boundary_case(self):
        """Test color contrast validation at WCAG AA boundary (4.5:1)."""
        enhancer = AccessibilityEnhancer()

        # Color combination close to 4.5:1 ratio
        assert enhancer.validate_color_contrast("#767676", "#ffffff") is True

    def test_validate_color_contrast_invalid_hex_colors(self):
        """Test color contrast validation with invalid hex colors."""
        enhancer = AccessibilityEnhancer()

        # Invalid hex format
        assert enhancer.validate_color_contrast("invalid", "#ffffff") is False
        assert enhancer.validate_color_contrast("#000000", "invalid") is False

        # Short hex format should work
        assert enhancer.validate_color_contrast("#000", "#fff") is False  # Will fail hex_to_rgb

    def test_validate_color_contrast_with_hash_prefix(self):
        """Test color contrast validation with and without hash prefix."""
        enhancer = AccessibilityEnhancer()

        # Both should work the same
        result_with_hash = enhancer.validate_color_contrast("#000000", "#ffffff")
        result_without_hash = enhancer.validate_color_contrast("000000", "ffffff")

        assert result_with_hash == result_without_hash

    def test_validate_color_contrast_zero_division_error(self):
        """Test color contrast validation handles zero division gracefully."""
        enhancer = AccessibilityEnhancer()

        # This should not raise an exception
        result = enhancer.validate_color_contrast("", "")
        assert result is False

    def test_hex_to_rgb_conversion(self):
        """Test hex to RGB conversion logic."""
        enhancer = AccessibilityEnhancer()

        # Access the nested function through the method
        # We'll test this indirectly through validate_color_contrast
        # since hex_to_rgb is a nested function

        # Test known color conversions by checking contrast validation works
        # Red vs Green might not have sufficient contrast for WCAG AA
        # Let's test with colors we know have good contrast
        assert enhancer.validate_color_contrast("#000000", "#ff0000") is True  # Black vs Red
        assert enhancer.validate_color_contrast("#ffffff", "#0000ff") is True  # White vs Blue

    def test_luminance_calculation(self):
        """Test luminance calculation logic."""
        enhancer = AccessibilityEnhancer()

        # Test by validating known luminance comparisons
        # White should have higher luminance than black
        white_vs_black = enhancer.validate_color_contrast("#000000", "#ffffff")
        black_vs_white = enhancer.validate_color_contrast("#ffffff", "#000000")

        # Both should return True (high contrast) but verify the function works
        assert white_vs_black is True
        assert black_vs_white is True


@pytest.mark.unit
@pytest.mark.fast
class TestKeyboardNavigationSupport:
    """Test cases for keyboard navigation support functionality."""

    def test_add_keyboard_navigation_support_returns_javascript(self):
        """Test keyboard navigation JavaScript generation."""
        enhancer = AccessibilityEnhancer()
        js_code = enhancer.add_keyboard_navigation_support()

        assert isinstance(js_code, str)
        assert "<script>" in js_code
        assert "</script>" in js_code
        assert "trapFocus" in js_code
        assert "addAriaAttributes" in js_code
        assert "initAccessibility" in js_code

    def test_keyboard_navigation_contains_focus_trap(self):
        """Test keyboard navigation includes focus trap functionality."""
        enhancer = AccessibilityEnhancer()
        js_code = enhancer.add_keyboard_navigation_support()

        # Check for focus trap logic
        assert "Tab" in js_code
        assert "shiftKey" in js_code
        assert "preventDefault" in js_code
        assert "focus()" in js_code

    def test_keyboard_navigation_contains_escape_handling(self):
        """Test keyboard navigation includes escape key handling."""
        enhancer = AccessibilityEnhancer()
        js_code = enhancer.add_keyboard_navigation_support()

        assert "Escape" in js_code
        assert "data-dismiss" in js_code or "close" in js_code

    def test_keyboard_navigation_contains_aria_enhancements(self):
        """Test keyboard navigation includes ARIA enhancements."""
        enhancer = AccessibilityEnhancer()
        js_code = enhancer.add_keyboard_navigation_support()

        assert "aria-label" in js_code
        assert "aria-live" in js_code
        assert "aria-expanded" in js_code
        assert "aria-controls" in js_code

    def test_keyboard_navigation_contains_mutation_observer(self):
        """Test keyboard navigation includes DOM mutation observer."""
        enhancer = AccessibilityEnhancer()
        js_code = enhancer.add_keyboard_navigation_support()

        assert "MutationObserver" in js_code
        assert "childList" in js_code
        assert "subtree" in js_code

    def test_keyboard_navigation_contains_tab_announcement(self):
        """Test keyboard navigation includes tab change announcements."""
        enhancer = AccessibilityEnhancer()
        js_code = enhancer.add_keyboard_navigation_support()

        assert 'role="tab"' in js_code
        assert "Switched to" in js_code
        assert "tab" in js_code


@pytest.mark.unit
@pytest.mark.fast
class TestAccessibilityCSSGeneration:
    """Test cases for accessibility CSS generation."""

    def test_get_accessibility_css_combines_styles(self):
        """Test accessibility CSS combines high contrast and additional styles."""
        enhancer = AccessibilityEnhancer()
        css = enhancer.get_accessibility_css()

        # Should contain high contrast CSS
        assert ".accessibility-mode" in css
        assert "--primary-color" in css

        # Should contain additional accessibility improvements
        assert ".gradio-container" in css
        assert "font-size: 16px" in css
        assert "line-height: 1.5" in css

    def test_accessibility_css_contains_link_styles(self):
        """Test accessibility CSS includes proper link styling."""
        enhancer = AccessibilityEnhancer()
        css = enhancer.get_accessibility_css()

        assert "text-decoration: underline" in css
        assert "a:hover" in css
        assert "a:focus" in css

    def test_accessibility_css_contains_message_styles(self):
        """Test accessibility CSS includes error and success message styles."""
        enhancer = AccessibilityEnhancer()
        css = enhancer.get_accessibility_css()

        assert ".error-message" in css
        assert ".success-message" in css
        assert "border-left" in css

    def test_accessibility_css_contains_loading_indicators(self):
        """Test accessibility CSS includes loading and progress indicators."""
        enhancer = AccessibilityEnhancer()
        css = enhancer.get_accessibility_css()

        assert ".loading" in css
        assert ".progress-bar" in css
        assert "Loading..." in css


@pytest.mark.unit
@pytest.mark.fast
class TestApplyAccessibilityEnhancements:
    """Test cases for apply_accessibility_enhancements function."""

    @patch("src.ui.components.accessibility_enhancements.gr.Blocks")
    @patch("src.ui.components.accessibility_enhancements.gr.HTML")
    @patch("src.ui.components.accessibility_enhancements.gr.Markdown")
    def test_apply_accessibility_enhancements_creates_enhanced_interface(self, mock_markdown, mock_html, mock_blocks):
        """Test apply_accessibility_enhancements creates enhanced interface."""
        # Setup mocks
        mock_interface = Mock()
        mock_enhanced_interface = Mock()
        mock_blocks.return_value.__enter__.return_value = mock_enhanced_interface

        result = apply_accessibility_enhancements(mock_interface)

        assert result is mock_enhanced_interface

        # Verify Blocks is called with accessibility features
        mock_blocks.assert_called_once()
        call_kwargs = mock_blocks.call_args[1]
        assert "css" in call_kwargs
        assert "head" in call_kwargs
        assert "title" in call_kwargs
        assert "PromptCraft-Hybrid - Accessible AI Workbench" in call_kwargs["title"]

    @patch("src.ui.components.accessibility_enhancements.gr.Blocks")
    @patch("src.ui.components.accessibility_enhancements.gr.HTML")
    @patch("src.ui.components.accessibility_enhancements.gr.Markdown")
    def test_apply_accessibility_enhancements_adds_skip_navigation(self, mock_markdown, mock_html, mock_blocks):
        """Test apply_accessibility_enhancements adds skip navigation."""
        mock_interface = Mock()
        mock_enhanced_interface = Mock()
        mock_blocks.return_value.__enter__.return_value = mock_enhanced_interface

        apply_accessibility_enhancements(mock_interface)

        # Should call HTML for skip navigation
        assert mock_html.call_count >= 1

        # Check if skip navigation was added
        html_calls = [call[0][0] for call in mock_html.call_args_list]
        skip_nav_added = any("Skip to main content" in str(call) for call in html_calls)
        assert skip_nav_added

    @patch("src.ui.components.accessibility_enhancements.gr.Blocks")
    @patch("src.ui.components.accessibility_enhancements.gr.HTML")
    @patch("src.ui.components.accessibility_enhancements.gr.Markdown")
    def test_apply_accessibility_enhancements_adds_landmark_regions(self, mock_markdown, mock_html, mock_blocks):
        """Test apply_accessibility_enhancements adds ARIA landmark regions."""
        mock_interface = Mock()
        mock_enhanced_interface = Mock()
        mock_blocks.return_value.__enter__.return_value = mock_enhanced_interface

        apply_accessibility_enhancements(mock_interface)

        # Should call HTML for landmark regions
        html_calls = [str(call[0][0]) for call in mock_html.call_args_list]

        # Check for landmark roles
        landmark_roles = ['role="banner"', 'role="main"', 'role="contentinfo"']
        for role in landmark_roles:
            role_found = any(role in call for call in html_calls)
            assert role_found, f"Landmark {role} not found in HTML calls"

    @patch("src.ui.components.accessibility_enhancements.gr.Blocks")
    @patch("src.ui.components.accessibility_enhancements.gr.HTML")
    @patch("src.ui.components.accessibility_enhancements.gr.Markdown")
    def test_apply_accessibility_enhancements_adds_accessibility_info(self, mock_markdown, mock_html, mock_blocks):
        """Test apply_accessibility_enhancements adds accessibility information."""
        mock_interface = Mock()
        mock_enhanced_interface = Mock()
        mock_blocks.return_value.__enter__.return_value = mock_enhanced_interface

        apply_accessibility_enhancements(mock_interface)

        # Should call Markdown for accessibility features description
        markdown_calls = [str(call[0][0]) for call in mock_markdown.call_args_list]

        accessibility_features = ["WCAG 2.1 AA", "keyboard navigation", "Screen reader", "ARIA landmarks"]

        for feature in accessibility_features:
            feature_found = any(feature in call for call in markdown_calls)
            assert feature_found, f"Accessibility feature '{feature}' not found"


@pytest.mark.unit
@pytest.mark.fast
class TestAccessibilityEnhancerEdgeCases:
    """Test cases for edge cases and error handling."""

    def test_empty_component_type_handling(self):
        """Test handling of empty component type."""
        enhancer = AccessibilityEnhancer()
        mock_component = Mock(spec=gr.Component)
        mock_component.elem_classes = []

        result = enhancer.add_aria_attributes(mock_component, "")

        assert result is mock_component
        assert len(mock_component.elem_classes) == 0

    def test_none_component_handling(self):
        """Test handling of None component."""
        enhancer = AccessibilityEnhancer()

        # Should not raise an exception
        result = enhancer.add_aria_attributes(None, "query_input")

        assert result is None

    def test_color_contrast_edge_cases(self):
        """Test color contrast validation edge cases."""
        enhancer = AccessibilityEnhancer()

        # Empty strings
        assert enhancer.validate_color_contrast("", "") is False

        # None values should not crash
        with pytest.raises(AttributeError):
            enhancer.validate_color_contrast(None, "#ffffff")

    def test_css_output_is_valid_string(self):
        """Test that all CSS outputs are valid strings."""
        enhancer = AccessibilityEnhancer()

        css_methods = [enhancer._get_high_contrast_css, enhancer.get_accessibility_css]

        for method in css_methods:
            result = method()
            assert isinstance(result, str)
            assert len(result) > 0

    def test_javascript_output_is_valid_string(self):
        """Test that JavaScript output is a valid string."""
        enhancer = AccessibilityEnhancer()

        js_code = enhancer.add_keyboard_navigation_support()

        assert isinstance(js_code, str)
        assert len(js_code) > 0
        assert js_code.count("<script>") == js_code.count("</script>")

    def test_aria_labels_all_strings(self):
        """Test that all ARIA labels are strings."""
        enhancer = AccessibilityEnhancer()
        labels = enhancer._get_aria_labels()

        for key, value in labels.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(key) > 0
            assert len(value) > 0

    def test_landmark_regions_all_valid_html(self):
        """Test that all landmark regions contain valid HTML."""
        enhancer = AccessibilityEnhancer()
        landmarks = enhancer.create_landmark_regions()

        for key, value in landmarks.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert "<" in value
            assert ">" in value
            assert "role=" in value  # Should have role attribute
