/*
*	Copyright (C) 2025 Kendall Tauser
*
*	This program is free software; you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation; either version 2 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License along
*	with this program; if not, write to the Free Software Foundation, Inc.,
*	51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    errors::LangExplorerError,
    grammar::{
        elem::GrammarElement,
        grammar::Grammar,
        lhs::ProductionLHS,
        prod::{context_free_production, production_rule, Production},
        rule::ProductionRule,
    },
    languages::{
        strings::{
            alphanumeric::{
                numeric_character_production_context_free, T_LO_A, T_LO_C, T_LO_E, T_LO_H, T_LO_I,
                T_LO_M, T_LO_N, T_LO_P, T_LO_R, T_LO_S, T_LO_T, T_LO_V, T_LO_W, T_LO_X,
            },
            nterminal_str, terminal_str, StringValue, COLON, COMMA, EPSILON, GREATER, LBRACKET,
            MINUS, PERCENT, PLUS, RBRACKET, SEMICOLON, SPACE,
        },
        GrammarBuilder,
    },
};

nterminal_str!(NT_ENTRYPOINT, "entrypoint");
nterminal_str!(NT_BLOCK, "block");
nterminal_str!(NT_SELECTOR, "selector");
nterminal_str!(NT_PROPERTIES, "properties");
nterminal_str!(NT_PROPERTY, "property");
nterminal_str!(NT_BASE_HTML_ELEMENT, "base_html_element");
nterminal_str!(NT_HTML_ELEMENT, "html_element");
nterminal_str!(NT_MULTI_CSHE, "comma_separated_html_elements");
nterminal_str!(NT_MULTI_SSHE, "space_separated_html_elements");
nterminal_str!(NT_PSEUDO_ELEMENTS, "pseudo_element");
nterminal_str!(NT_PSEUDO_CLASSES, "pseudo_class");
nterminal_str!(NT_CLASS, "class");
nterminal_str!(NT_ID, "id");
nterminal_str!(NT_COLOR, "color");
nterminal_str!(NT_DISPLAY_TYPE, "display_type");
nterminal_str!(NT_ALIGN_CONTENT_TYPE, "align_content_type");
nterminal_str!(NT_ALIGN_ITEMS_TYPE, "align_items_type");
nterminal_str!(NT_ALIGN_SELF_TYPE, "align_self_type");
nterminal_str!(NT_FLOAT_TYPE, "float_type");
nterminal_str!(NT_JUSTIFY_CONTENT_TYPE, "justify_content_type");
nterminal_str!(NT_JUSTIFY_ITEMS_TYPE, "justify_items_type");
nterminal_str!(NT_JUSTIFY_SELF_TYPE, "justify_self_type");
nterminal_str!(NT_SIZE, "size");
nterminal_str!(NT_NUMBER, "number");
nterminal_str!(NT_SIZE_SUFFIX, "size_suffix");
nterminal_str!(NT_BORDER_TYPE, "border_type");
nterminal_str!(NT_OVERFLOW_TYPE, "overflow_type");
nterminal_str!(NT_TEXT_ALIGN_TYPE, "text_align_type");

// Enumeration of properties
terminal_str!(ALIGN_CONTENT, "align-content");
terminal_str!(ALIGN_ITEMS, "align-items");
terminal_str!(ALIGN_SELF, "align-self");
terminal_str!(ALL, "all");
terminal_str!(ANIMATION, "animation");
terminal_str!(ANIMATION_DELAY, "animation-delay");
terminal_str!(ANIMATION_DIRECTION, "animation-direction");
terminal_str!(ANIMATION_DURATION, "animation-duration");
terminal_str!(ANIMATION_FILL_MODE, "animation-fill-mode");
terminal_str!(ANIMATION_ITERATION_COUNT, "animation-iteration-count");
terminal_str!(ANIMATION_NAME, "animation-name");
terminal_str!(ANIMATION_PLAY_STATE, "animation-play-state");
terminal_str!(ANIMATION_TIMING_FUNCTION, "animation-timing-function");
terminal_str!(BACKFACE_VISIBILITY, "backface-visibility");
terminal_str!(BACKGROUND, "background");
terminal_str!(BACKGROUND_ATTACHMENT, "background-attachment");
terminal_str!(BACKGROUND_BLEND_MODE, "background-blend-mode");
terminal_str!(BACKGROUND_CLIP, "background-clip");
terminal_str!(BACKGROUND_COLOR, "background-color");
terminal_str!(BACKGROUND_IMAGE, "background-image");
terminal_str!(BACKGROUND_ORIGIN, "background-origin");
terminal_str!(BACKGROUND_POSITION, "background-position");
terminal_str!(BACKGROUND_REPEAT, "background-repeat");
terminal_str!(BACKGROUND_SIZE, "background-size");
terminal_str!(BORDER, "border");
terminal_str!(BORDER_BOTTOM, "border-bottom");
terminal_str!(BORDER_BOTTOM_COLOR, "border-bottom-color");
terminal_str!(BORDER_BOTTOM_LEFT_RADIUS, "border-bottom-left-radius");
terminal_str!(BORDER_BOTTOM_RIGHT_RADIUS, "border-bottom-right-radius");
terminal_str!(BORDER_BOTTOM_STYLE, "border-bottom-style");
terminal_str!(BORDER_BOTTOM_WIDTH, "border-bottom-width");
terminal_str!(BORDER_COLLAPSE, "border-collapse");
terminal_str!(BORDER_COLOR, "border-color");
terminal_str!(BORDER_IMAGE, "border-image");
terminal_str!(BORDER_IMAGE_OUTSET, "border-image-outset");
terminal_str!(BORDER_IMAGE_REPEAT, "border-image-repeat");
terminal_str!(BORDER_IMAGE_SLICE, "border-image-slice");
terminal_str!(BORDER_IMAGE_SOURCE, "border-image-source");
terminal_str!(BORDER_IMAGE_WIDTH, "border-image-width");
terminal_str!(BORDER_LEFT, "border-left");
terminal_str!(BORDER_LEFT_COLOR, "border-left-color");
terminal_str!(BORDER_LEFT_STYLE, "border-left-style");
terminal_str!(BORDER_LEFT_WIDTH, "border-left-width");
terminal_str!(BORDER_RADIUS, "border-radius");
terminal_str!(BORDER_RIGHT, "border-right");
terminal_str!(BORDER_RIGHT_COLOR, "border-right-color");
terminal_str!(BORDER_RIGHT_STYLE, "border-right-style");
terminal_str!(BORDER_RIGHT_WIDTH, "border-right-width");
terminal_str!(BORDER_SPACING, "border-spacing");
terminal_str!(BORDER_STYLE, "border-style");
terminal_str!(BORDER_TOP, "border-top");
terminal_str!(BORDER_TOP_COLOR, "border-top-color");
terminal_str!(BORDER_TOP_LEFT_RADIUS, "border-top-left-radius");
terminal_str!(BORDER_TOP_RIGHT_RADIUS, "border-top-right-radius");
terminal_str!(BORDER_TOP_STYLE, "border-top-style");
terminal_str!(BORDER_TOP_WIDTH, "border-top-width");
terminal_str!(BORDER_WIDTH, "border-width");
terminal_str!(BOTTOM, "bottom");
terminal_str!(BOX_SHADOW, "box-shadow");
terminal_str!(BOX_SIZING, "box-sizing");
terminal_str!(CAPTION_SIDE, "caption-side");
terminal_str!(CLEAR, "clear");
terminal_str!(CLIP, "clip");
terminal_str!(COLOR, "color");
terminal_str!(COLUMN_COUNT, "column-count");
terminal_str!(COLUMN_FILL, "column-fill");
terminal_str!(COLUMN_GAP, "column-gap");
terminal_str!(COLUMN_RULE, "column-rule");
terminal_str!(COLUMN_RULE_COLOR, "column-rule-color");
terminal_str!(COLUMN_RULE_STYLE, "column-rule-style");
terminal_str!(COLUMN_RULE_WIDTH, "column-rule-width");
terminal_str!(COLUMN_SPAN, "column-span");
terminal_str!(COLUMN_WIDTH, "column-width");
terminal_str!(COLUMNS, "columns");
terminal_str!(CONTENT, "content");
terminal_str!(COUNTER_INCREMENT, "counter-increment");
terminal_str!(COUNTER_RESET, "counter-reset");
terminal_str!(CURSOR, "cursor");
terminal_str!(DIRECTION, "direction");
terminal_str!(DISPLAY, "display");
terminal_str!(EMPTY_CELLS, "empty-cells");
terminal_str!(FILTER, "filter");
terminal_str!(FLEX, "flex");
terminal_str!(FLEX_BASIS, "flex-basis");
terminal_str!(FLEX_DIRECTION, "flex-direction");
terminal_str!(FLEX_FLOW, "flex-flow");
terminal_str!(FLEX_GROW, "flex-grow");
terminal_str!(FLEX_SHRINK, "flex-shrink");
terminal_str!(FLEX_WRAP, "flex-wrap");
terminal_str!(FLOAT, "float");
terminal_str!(FONT, "font");
terminal_str!(FONT_FACE, "@font-face");
terminal_str!(FONT_FAMILY, "font-family");
terminal_str!(FONT_SIZE, "font-size");
terminal_str!(FONT_SIZE_ADJUST, "font-size-adjust");
terminal_str!(FONT_STRETCH, "font-stretch");
terminal_str!(FONT_STYLE, "font-style");
terminal_str!(FONT_VARIANT, "font-variant");
terminal_str!(FONT_WEIGHT, "font-weight");
terminal_str!(HANGING_PUNCTUATION, "hanging-punctuation");
terminal_str!(HEIGHT, "height");
terminal_str!(JUSTIFY_CONTENT, "justify-content");
terminal_str!(JUSTIFY_ITEMS, "justify-items");
terminal_str!(JUSTIFY_SELF, "justify-self");
terminal_str!(KEYFRAMES, "@keyframes");
terminal_str!(LEFT, "left");
terminal_str!(LETTER_SPACING, "letter-spacing");
terminal_str!(LINE_HEIGHT, "line-height");
terminal_str!(LIST_STYLE, "list-style");
terminal_str!(LIST_STYLE_IMAGE, "list-style-image");
terminal_str!(LIST_STYLE_POSITION, "list-style-position");
terminal_str!(LIST_STYLE_TYPE, "list-style-type");
terminal_str!(MARGIN, "margin");
terminal_str!(MARGIN_BOTTOM, "margin-bottom");
terminal_str!(MARGIN_LEFT, "margin-left");
terminal_str!(MARGIN_RIGHT, "margin-right");
terminal_str!(MARGIN_TOP, "margin-top");
terminal_str!(MAX_HEIGHT, "max-height");
terminal_str!(MAX_WIDTH, "max-width");
terminal_str!(MEDIA, "@media");
terminal_str!(MIN_HEIGHT, "min-height");
terminal_str!(MIN_WIDTH, "min-width");
terminal_str!(NAV_DOWN, "nav-down");
terminal_str!(NAV_INDEX, "nav-index");
terminal_str!(NAV_LEFT, "nav-left");
terminal_str!(NAV_RIGHT, "nav-right");
terminal_str!(NAV_UP, "nav-up");
terminal_str!(OPACITY, "opacity");
terminal_str!(ORDER, "order");
terminal_str!(OUTLINE, "outline");
terminal_str!(OUTLINE_COLOR, "outline-color");
terminal_str!(OUTLINE_OFFSET, "outline-offset");
terminal_str!(OUTLINE_STYLE, "outline-style");
terminal_str!(OUTLINE_WIDTH, "outline-width");
terminal_str!(OVERFLOW, "overflow");
terminal_str!(OVERFLOW_X, "overflow-x");
terminal_str!(OVERFLOW_Y, "overflow-y");
terminal_str!(PADDING, "padding");
terminal_str!(PADDING_BOTTOM, "padding-bottom");
terminal_str!(PADDING_LEFT, "padding-left");
terminal_str!(PADDING_RIGHT, "padding-right");
terminal_str!(PADDING_TOP, "padding-top");
terminal_str!(PAGE_BREAK_AFTER, "page-break-after");
terminal_str!(PAGE_BREAK_BEFORE, "page-break-before");
terminal_str!(PAGE_BREAK_INSIDE, "page-break-inside");
terminal_str!(PERSPECTIVE, "perspective");
terminal_str!(PERSPECTIVE_ORIGIN, "perspective-origin");
terminal_str!(POSITION, "position");
terminal_str!(QUOTES, "quotes");
terminal_str!(RESIZE, "resize");
terminal_str!(RIGHT, "right");
terminal_str!(SCROLL_BEHAVIOR, "scroll-behavior");
terminal_str!(SCROLL_MARGIN, "scroll-margin");
terminal_str!(SCROLL_MARGIN_BLOCK, "scroll-margin-block");
terminal_str!(SCROLL_MARGIN_BLOCK_END, "scroll-margin-block-end");
terminal_str!(SCROLL_MARGIN_BLOCK_START, "scroll-margin-block-start");
terminal_str!(SCROLL_MARGIN_BOTTOM, "scroll-margin-bottom");
terminal_str!(SCROLL_MARGIN_INLINE, "scroll-margin-inline");
terminal_str!(SCROLL_MARGIN_INLINE_END, "scroll-margin-inline-end");
terminal_str!(SCROLL_MARGIN_INLINE_START, "scroll-margin-inline-start");
terminal_str!(SCROLL_MARGIN_LEFT, "scroll-margin-left");
terminal_str!(SCROLL_MARGIN_RIGHT, "scroll-margin-right");
terminal_str!(SCROLL_MARGIN_TOP, "scroll-margin-top");
terminal_str!(SCROLL_PADDING, "scroll-padding");
terminal_str!(SCROLL_PADDING_BLOCK, "scroll-padding-block");
terminal_str!(SCROLL_PADDING_BLOCK_END, "scroll-padding-block-end");
terminal_str!(SCROLL_PADDING_BLOCK_START, "scroll-padding-block-start");
terminal_str!(SCROLL_PADDING_BOTTOM, "scroll-padding-bottom");
terminal_str!(SCROLL_PADDING_INLINE, "scroll-padding-inline");
terminal_str!(SCROLL_PADDING_INLINE_END, "scroll-padding-inline-end");
terminal_str!(SCROLL_PADDING_INLINE_START, "scroll-padding-inline-start");
terminal_str!(SCROLL_PADDING_LEFT, "scroll-padding-left");
terminal_str!(SCROLL_PADDING_RIGHT, "scroll-padding-right");
terminal_str!(SCROLL_PADDING_TOP, "scroll-padding-top");
terminal_str!(TAB_SIZE, "tab-size");
terminal_str!(TABLE_LAYOUT, "table-layout");
terminal_str!(TEXT_ALIGN, "text-align");
terminal_str!(TEXT_ALIGN_LAST, "text-align-last");
terminal_str!(TEXT_DECORATION, "text-decoration");
terminal_str!(TEXT_INDENT, "text-indent");
terminal_str!(TEXT_OVERFLOW, "text-overflow");
terminal_str!(TEXT_SHADOW, "text-shadow");
terminal_str!(TEXT_TRANSFORM, "text-transform");
terminal_str!(TOP, "top");
terminal_str!(TRANSFORM, "transform");
terminal_str!(TRANSFORM_ORIGIN, "transform-origin");
terminal_str!(TRANSFORM_STYLE, "transform-style");
terminal_str!(TRANSITION, "transition");
terminal_str!(TRANSITION_DELAY, "transition-delay");
terminal_str!(TRANSITION_DURATION, "transition-duration");
terminal_str!(TRANSITION_PROPERTY, "transition-property");
terminal_str!(TRANSITION_TIMING_FUNCTION, "transition-timing-function");
terminal_str!(UNICODE_BIDI, "unicode-bidi");
terminal_str!(USER_SELECT, "user-select");
terminal_str!(VERTICAL_ALIGN, "vertical-align");
terminal_str!(VISIBILITY, "visibility");
terminal_str!(WHITE_SPACE, "white-space");
terminal_str!(WIDTH, "width");
terminal_str!(WORD_BREAK, "word-break");
terminal_str!(WIDOWS, "widows");
terminal_str!(WORD_SPACING, "word-spacing");
terminal_str!(WORD_WRAP, "word-wrap");
terminal_str!(Z_INDEX, "z-index");

// Enumeration of all HTML elements
terminal_str!(A, "a");
terminal_str!(ABBR, "abbr");
terminal_str!(ADDRESS, "address");
terminal_str!(AREA, "area");
terminal_str!(ARTICLE, "article");
terminal_str!(ASIDE, "aside");
terminal_str!(AUDIO, "audio");
terminal_str!(B, "b");
terminal_str!(BASE, "base");
terminal_str!(BDI, "bdi");
terminal_str!(BDO, "bdo");
terminal_str!(BLOCKQUOTE, "blockquote");
terminal_str!(BODY, "body");
terminal_str!(BR, "br");
terminal_str!(BUTTON, "button");
terminal_str!(CANVAS, "canvas");
terminal_str!(CAPTION, "caption");
terminal_str!(CITE, "cite");
terminal_str!(CODE, "code");
terminal_str!(COL, "col");
terminal_str!(COLGROUP, "colgroup");
terminal_str!(DATA, "data");
terminal_str!(DATALIST, "datalist");
terminal_str!(DD, "dd");
terminal_str!(DEL, "del");
terminal_str!(DETAILS, "details");
terminal_str!(DFN, "dfn");
terminal_str!(DIALOG, "dialog");
terminal_str!(DIV, "div");
terminal_str!(DL, "dl");
terminal_str!(DT, "dt");
terminal_str!(EM, "em");
terminal_str!(EMBED, "embed");
terminal_str!(FIELDSET, "fieldset");
terminal_str!(FIGCAPTION, "figcaption");
terminal_str!(FIGURE, "figure");
terminal_str!(FOOTER, "footer");
terminal_str!(FORM, "form");
terminal_str!(H1, "h1");
terminal_str!(H2, "h2");
terminal_str!(H3, "h3");
terminal_str!(H4, "h4");
terminal_str!(H5, "h5");
terminal_str!(H6, "h6");
terminal_str!(HEAD, "head");
terminal_str!(HEADER, "header");
terminal_str!(HGROUP, "hgroup");
terminal_str!(HR, "hr");
terminal_str!(HTML, "html");
terminal_str!(I, "i");
terminal_str!(IFRAME, "iframe");
terminal_str!(IMG, "img");
terminal_str!(INPUT, "input");
terminal_str!(INS, "ins");
terminal_str!(KBD, "kbd");
terminal_str!(LABEL, "label");
terminal_str!(LEGEND, "legend");
terminal_str!(LI, "li");
terminal_str!(LINK, "link");
terminal_str!(MAIN, "main");
terminal_str!(MAP, "map");
terminal_str!(MARK, "mark");
terminal_str!(META, "meta");
terminal_str!(METER, "meter");
terminal_str!(NAV, "nav");
terminal_str!(NOSCRIPT, "noscript");
terminal_str!(OBJECT, "object");
terminal_str!(OL, "ol");
terminal_str!(OPTGROUP, "optgroup");
terminal_str!(OPTION, "option");
terminal_str!(OUTPUT, "output");
terminal_str!(P, "p");
terminal_str!(PARAM, "param");
terminal_str!(PICTURE, "picture");
terminal_str!(PRE, "pre");
terminal_str!(PROGRESS, "progress");
terminal_str!(Q, "q");
terminal_str!(RP, "rp");
terminal_str!(RT, "rt");
terminal_str!(RUBY, "ruby");
terminal_str!(S, "s");
terminal_str!(SAMP, "samp");
terminal_str!(SCRIPT, "script");
terminal_str!(SECTION, "section");
terminal_str!(SELECT, "select");
terminal_str!(SMALL, "small");
terminal_str!(SOURCE, "source");
terminal_str!(SPAN, "span");
terminal_str!(STRONG, "strong");
terminal_str!(STYLE, "style");
terminal_str!(SUB, "sub");
terminal_str!(SUMMARY, "summary");
terminal_str!(SUP, "sup");
terminal_str!(TABLE, "table");
terminal_str!(TBODY, "tbody");
terminal_str!(TD, "td");
terminal_str!(TEMPLATE, "template");
terminal_str!(TEXTAREA, "textarea");
terminal_str!(TFOOT, "tfoot");
terminal_str!(TH, "th");
terminal_str!(THEAD, "thead");
terminal_str!(TIME, "time");
terminal_str!(TITLE, "title");
terminal_str!(TR, "tr");
terminal_str!(TRACK, "track");
terminal_str!(U, "u");
terminal_str!(UL, "ul");
terminal_str!(VAR, "var");
terminal_str!(VIDEO, "video");
terminal_str!(WBR, "wbr");

// Enumeration of pseudo-classes
// LINK should be included here, but it is also an HTML element, so it is included above.
terminal_str!(VISITED, "visited");
terminal_str!(HOVER, "hover");
terminal_str!(ACTIVE, "active");

// Enumeration of pseudo-elements
terminal_str!(FIRST_LETTER, "first-letter");
terminal_str!(FIRST_LINE, "first-line");
terminal_str!(MARKER, "marker");
terminal_str!(BEFORE, "before");
terminal_str!(AFTER, "after");
terminal_str!(SELECTION, "selection");

// Enumeration of display types.
terminal_str!(INLINE, "inline");
terminal_str!(BLOCK, "block");
terminal_str!(CONTENTS, "contents");
// terminal_str!(FLEX, "flex"); // This is already a property.
terminal_str!(GRID, "grid");
terminal_str!(INLINE_BLOCK, "inline-block");
terminal_str!(INLINE_FLEX, "inline-flex");
terminal_str!(INLINE_GRID, "inline-grid");
terminal_str!(INLINE_TABLE, "inline-table");
terminal_str!(LIST_ITEM, "list-item");
terminal_str!(RUN_IN, "run-in");
// terminal_str!(TABLE, "table"); // This is already a property.
terminal_str!(TABLE_CAPTION, "table-caption");
terminal_str!(TABLE_COLUMN_GROUP, "table-column-group");
terminal_str!(TABLE_HEADER_GROUP, "table-header-group");
terminal_str!(TABLE_FOOTER_GROUP, "table-footer-group");
terminal_str!(TABLE_ROW_GROUP, "table-row-group");
terminal_str!(TABLE_CELL, "table-cell");
terminal_str!(TABLE_COLUMN, "table-column");
terminal_str!(TABLE_ROW, "table-row");
terminal_str!(NONE, "none");
terminal_str!(INITIAL, "initial");
terminal_str!(INHERIT, "inherit");

// Enumeration of align-content options
terminal_str!(STRETCH, "stretch");
terminal_str!(CENTER, "center");
terminal_str!(FLEX_START, "flex-start");
terminal_str!(FLEX_END, "flex-end");
terminal_str!(SPACE_BETWEEN, "space-between");
terminal_str!(SPACE_AROUND, "space-around");
terminal_str!(SPACE_EVENLY, "space-evenly");
// terminal_str!(INHERIT, "inherit");

// Enumeration of align-items options
terminal_str!(NORMAL, "normal");
// terminal_str!(STRETCH, "stretch");
// terminal_str!(CENTER, "center");
// terminal_str!(FLEX_START, "flex-start");
// terminal_str!(FLEX_END, "flex-end");
terminal_str!(START, "start");
terminal_str!(END, "end");
terminal_str!(BASELINE, "baseline");
// terminal_str!(INITIAL, "initial");
// terminal_str!(INHERIT, "inherit");

// Enumeration of align-self items
terminal_str!(AUTO, "auto");
// terminal_str!(STRETCH, "stretch");
// terminal_str!(CENTER, "center");
// terminal_str!(FLEX_START, "flex-start");
// terminal_str!(FLEX_END, "flex-end");
// terminal_str!(BASELINE, "baseline");
// terminal_str!(INITIAL, "initial");
// terminal_str!(INHERIT, "inherit");

// Enumeration of border styles
// terminal_str!(NONE, "none");
terminal_str!(HIDDEN, "hidden");
terminal_str!(DOTTED, "dotted");
terminal_str!(DASHED, "dashed");
terminal_str!(SOLID, "solid");
terminal_str!(DOUBLE, "double");
terminal_str!(GROOVE, "groove");
terminal_str!(RIDGE, "ridge");
terminal_str!(INSET, "inset");
terminal_str!(OUTSET, "outset");
// terminal_str!(INITIAL, "initial");
// terminal_str!(INHERIT, "inherit");

// Enumeration of overflow items
terminal_str!(VISIBLE, "visible");
// terminal_str!(HIDDEN, "hidden");
// terminal_str!(CLIP, "clip");
terminal_str!(SCROLL, "scroll");
// terminal_str!(AUTO, "auto");
// terminal_str!(INITIAL, "initial");
// terminal_str!(INHERIT, "inherit");

// Enumeration of text-align items
// terminal_str!(LEFT, "left");
// terminal_str!(RIGHT, "right");s
terminal_str!(JUSTIFY, "justify");
// terminal_str!(INITIAL, "initial");
// terminal_str!(INHERIT, "inherit");

pub struct CSSLanguage;

/// Parameters for CSS Language.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CSSLanguageParameters {
    #[serde(alias = "version")]
    version: CSSLanguageVersion,

    colors: Vec<String>,
    classes: Vec<String>,
    ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum CSSLanguageVersion {
    /// Includes as much of the CSS language as I am able to reasonbaly represent in
    /// my limited existence.
    ExhaustiveV1,
}

impl Default for CSSLanguageVersion {
    fn default() -> Self {
        Self::ExhaustiveV1
    }
}

impl Default for CSSLanguageParameters {
    fn default() -> Self {
        Self {
            version: Default::default(),
            colors: vec![
                "#842d5b".into(),
                "#20b01c".into(),
                "#7d1dc1".into(),
                "#42a1dc".into(),
                "#da8454".into(),
                "#8ec5d2".into(),
                "#a69657".into(),
                "#a69657".into(),
                "#664ba3".into(),
                "#3b6a42".into(),
                "rgb(39, 37, 193)".into(),
                "rgb(37, 138, 166)".into(),
                "rgb(84, 183, 126)".into(),
                "rgb(104, 36, 170)".into(),
                "rgb(207, 106, 144)".into(),
                "rgb(203, 135, 198)".into(),
                "rgb(231, 100, 187)".into(),
                "rgb(143, 143, 143)".into(),
                "rgb(68, 68, 68)".into(),
                "rgb(160, 160, 160)".into(),
                "rgb(141, 141, 141)".into(),
                "rgb(95, 95, 95)".into(),
            ],
            classes: vec!["buttons".into(), "footers".into()],
            ids: vec!["content".into(), "header".into(), "btn".into()],
        }
    }
}

impl GrammarBuilder for CSSLanguage {
    type Term = StringValue;
    type NTerm = StringValue;
    type Params<'de> = CSSLanguageParameters;

    fn generate_grammar<'de>(
        params: Self::Params<'de>,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError> {
        let mut classes: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        for var in params.classes.iter() {
            // Store this variable in the heap.
            let term = GrammarElement::Terminal(var.into());
            classes.push(production_rule!(term));
        }

        let mut ids: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        for var in params.ids.iter() {
            // Store this variable in the heap.
            let term = GrammarElement::Terminal(var.into());
            ids.push(production_rule!(term));
        }

        let mut colors: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        for var in params.colors.iter() {
            // Store this variable in the heap.
            let term = GrammarElement::Terminal(var.into());
            colors.push(production_rule!(term));
        }

        let grammar = Grammar::new(
            "entrypoint".into(),
            vec![
                context_free_production!(
                    NT_ENTRYPOINT,
                    // Optionally create no blocks
                    production_rule!(2, EPSILON),
                    // Or generate block
                    production_rule!(40, NT_BLOCK),
                    // Or generate multiple blocks
                    production_rule!(58, NT_BLOCK, NT_ENTRYPOINT)
                ),
                context_free_production!(
                    NT_BLOCK,
                    production_rule!(NT_SELECTOR, LBRACKET, NT_PROPERTIES, RBRACKET)
                ),
                context_free_production!(
                    NT_MULTI_CSHE,
                    production_rule!(NT_BASE_HTML_ELEMENT),
                    // Imperfect, but whatever for right now
                    production_rule!(NT_BASE_HTML_ELEMENT, COMMA, NT_MULTI_CSHE)
                ),
                context_free_production!(
                    NT_MULTI_SSHE,
                    production_rule!(NT_BASE_HTML_ELEMENT),
                    // Imperfect, but whatever for right now
                    production_rule!(NT_BASE_HTML_ELEMENT, SPACE, NT_MULTI_SSHE)
                ),
                context_free_production!(
                    NT_SELECTOR,
                    // Only one HTML element
                    production_rule!(50, NT_BASE_HTML_ELEMENT),
                    // All elements of specified one level deep from parent element type
                    production_rule!(10, NT_BASE_HTML_ELEMENT, GREATER, NT_BASE_HTML_ELEMENT),
                    production_rule!(10, NT_BASE_HTML_ELEMENT, PLUS, NT_BASE_HTML_ELEMENT),
                    production_rule!(10, NT_BASE_HTML_ELEMENT, MINUS, NT_BASE_HTML_ELEMENT),
                    production_rule!(10, NT_MULTI_CSHE),
                    production_rule!(10, NT_MULTI_SSHE)
                ),
                context_free_production!(
                    NT_PROPERTIES,
                    // Optionally no properties
                    production_rule!(30, EPSILON),
                    // Or one property
                    production_rule!(40, NT_PROPERTY),
                    // Or many properties
                    production_rule!(40, NT_PROPERTY, NT_PROPERTY)
                ),
                context_free_production!(
                    NT_PROPERTY,
                    production_rule!(ALIGN_CONTENT, COLON, NT_ALIGN_CONTENT_TYPE, SEMICOLON),
                    production_rule!(ALIGN_ITEMS, COLON, NT_ALIGN_ITEMS_TYPE, SEMICOLON),
                    production_rule!(ALIGN_SELF, COLON, NT_ALIGN_SELF_TYPE, SEMICOLON),
                    production_rule!(ALL, COLON, SEMICOLON),
                    production_rule!(ANIMATION, COLON, SEMICOLON),
                    production_rule!(ANIMATION_DELAY, COLON, SEMICOLON),
                    production_rule!(ANIMATION_DIRECTION, COLON, SEMICOLON),
                    production_rule!(ANIMATION_DURATION, COLON, SEMICOLON),
                    production_rule!(ANIMATION_FILL_MODE, COLON, SEMICOLON),
                    production_rule!(ANIMATION_ITERATION_COUNT, COLON, SEMICOLON),
                    production_rule!(ANIMATION_NAME, COLON, SEMICOLON),
                    production_rule!(ANIMATION_PLAY_STATE, COLON, SEMICOLON),
                    production_rule!(ANIMATION_TIMING_FUNCTION, COLON, SEMICOLON),
                    production_rule!(BACKFACE_VISIBILITY, COLON, SEMICOLON),
                    production_rule!(BACKGROUND, COLON, NT_COLOR, SEMICOLON),
                    production_rule!(BACKGROUND_ATTACHMENT, COLON, SEMICOLON),
                    production_rule!(BACKGROUND_BLEND_MODE, COLON, SEMICOLON),
                    production_rule!(BACKGROUND_CLIP, COLON, SEMICOLON),
                    production_rule!(BACKGROUND_COLOR, COLON, NT_COLOR, SEMICOLON),
                    production_rule!(BACKGROUND_IMAGE, COLON, SEMICOLON),
                    production_rule!(BACKGROUND_ORIGIN, COLON, SEMICOLON),
                    production_rule!(BACKGROUND_POSITION, COLON, SEMICOLON),
                    production_rule!(BACKGROUND_REPEAT, COLON, SEMICOLON),
                    production_rule!(BACKGROUND_SIZE, COLON, SEMICOLON),
                    production_rule!(BORDER, COLON, SEMICOLON),
                    production_rule!(
                        BORDER_BOTTOM,
                        COLON,
                        NT_SIZE,
                        NT_BORDER_TYPE,
                        NT_COLOR,
                        SEMICOLON
                    ),
                    production_rule!(BORDER_BOTTOM_COLOR, COLON, NT_COLOR, SEMICOLON),
                    production_rule!(BORDER_BOTTOM_LEFT_RADIUS, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(BORDER_BOTTOM_RIGHT_RADIUS, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(BORDER_BOTTOM_STYLE, COLON, NT_BORDER_TYPE, SEMICOLON),
                    production_rule!(BORDER_BOTTOM_WIDTH, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(BORDER_COLLAPSE, COLON, SEMICOLON),
                    production_rule!(BORDER_COLOR, COLON, NT_COLOR, SEMICOLON),
                    production_rule!(BORDER_IMAGE, COLON, SEMICOLON),
                    production_rule!(BORDER_IMAGE_OUTSET, COLON, SEMICOLON),
                    production_rule!(BORDER_IMAGE_REPEAT, COLON, SEMICOLON),
                    production_rule!(BORDER_IMAGE_SLICE, COLON, SEMICOLON),
                    production_rule!(BORDER_IMAGE_SOURCE, COLON, SEMICOLON),
                    production_rule!(BORDER_IMAGE_WIDTH, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(
                        BORDER_LEFT,
                        COLON,
                        NT_SIZE,
                        NT_BORDER_TYPE,
                        NT_COLOR,
                        SEMICOLON
                    ),
                    production_rule!(BORDER_LEFT_COLOR, COLON, NT_COLOR, SEMICOLON),
                    production_rule!(BORDER_LEFT_STYLE, COLON, NT_BORDER_TYPE, SEMICOLON),
                    production_rule!(BORDER_LEFT_WIDTH, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(BORDER_RADIUS, COLON, SEMICOLON),
                    production_rule!(
                        BORDER_RIGHT,
                        COLON,
                        NT_SIZE,
                        NT_BORDER_TYPE,
                        NT_COLOR,
                        SEMICOLON
                    ),
                    production_rule!(BORDER_RIGHT_COLOR, COLON, NT_COLOR, SEMICOLON),
                    production_rule!(BORDER_RIGHT_STYLE, COLON, NT_BORDER_TYPE, SEMICOLON),
                    production_rule!(BORDER_RIGHT_WIDTH, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(BORDER_SPACING, COLON, SEMICOLON),
                    production_rule!(BORDER_STYLE, COLON, SEMICOLON),
                    production_rule!(
                        BORDER_TOP,
                        COLON,
                        NT_SIZE,
                        NT_BORDER_TYPE,
                        NT_COLOR,
                        SEMICOLON
                    ),
                    production_rule!(BORDER_TOP_COLOR, COLON, NT_COLOR, SEMICOLON),
                    production_rule!(BORDER_TOP_LEFT_RADIUS, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(BORDER_TOP_RIGHT_RADIUS, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(BORDER_TOP_STYLE, COLON, NT_BORDER_TYPE, SEMICOLON),
                    production_rule!(BORDER_TOP_WIDTH, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(BORDER_WIDTH, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(BOTTOM, COLON, SEMICOLON),
                    production_rule!(BOX_SHADOW, COLON, SEMICOLON),
                    production_rule!(BOX_SIZING, COLON, SEMICOLON),
                    production_rule!(CAPTION_SIDE, COLON, SEMICOLON),
                    production_rule!(CLEAR, COLON, SEMICOLON),
                    production_rule!(CLIP, COLON, SEMICOLON),
                    production_rule!(COLOR, COLON, NT_COLOR, SEMICOLON),
                    production_rule!(COLUMN_COUNT, COLON, SEMICOLON),
                    production_rule!(COLUMN_FILL, COLON, SEMICOLON),
                    production_rule!(COLUMN_GAP, COLON, SEMICOLON),
                    production_rule!(COLUMN_RULE, COLON, SEMICOLON),
                    production_rule!(COLUMN_RULE_COLOR, COLON, SEMICOLON),
                    production_rule!(COLUMN_RULE_STYLE, COLON, SEMICOLON),
                    production_rule!(COLUMN_RULE_WIDTH, COLON, SEMICOLON),
                    production_rule!(COLUMN_SPAN, COLON, SEMICOLON),
                    production_rule!(COLUMN_WIDTH, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(COLUMNS, COLON, SEMICOLON),
                    production_rule!(CONTENT, COLON, SEMICOLON),
                    production_rule!(COUNTER_INCREMENT, COLON, SEMICOLON),
                    production_rule!(COUNTER_RESET, COLON, SEMICOLON),
                    production_rule!(CURSOR, COLON, SEMICOLON),
                    production_rule!(DIRECTION, COLON, SEMICOLON),
                    production_rule!(DISPLAY, COLON, NT_DISPLAY_TYPE, SEMICOLON),
                    production_rule!(EMPTY_CELLS, COLON, SEMICOLON),
                    production_rule!(FILTER, COLON, SEMICOLON),
                    production_rule!(FLEX, COLON, SEMICOLON),
                    production_rule!(FLEX_BASIS, COLON, SEMICOLON),
                    production_rule!(FLEX_DIRECTION, COLON, SEMICOLON),
                    production_rule!(FLEX_FLOW, COLON, SEMICOLON),
                    production_rule!(FLEX_GROW, COLON, SEMICOLON),
                    production_rule!(FLEX_SHRINK, COLON, SEMICOLON),
                    production_rule!(FLEX_WRAP, COLON, SEMICOLON),
                    production_rule!(FLOAT, COLON, NT_FLOAT_TYPE, SEMICOLON),
                    production_rule!(FONT, COLON, SEMICOLON),
                    production_rule!(FONT_FACE, COLON, SEMICOLON),
                    production_rule!(FONT_FAMILY, COLON, SEMICOLON),
                    production_rule!(FONT_SIZE, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(FONT_SIZE_ADJUST, COLON, SEMICOLON),
                    production_rule!(FONT_STRETCH, COLON, SEMICOLON),
                    production_rule!(FONT_STYLE, COLON, SEMICOLON),
                    production_rule!(FONT_VARIANT, COLON, SEMICOLON),
                    production_rule!(FONT_WEIGHT, COLON, SEMICOLON),
                    production_rule!(HANGING_PUNCTUATION, COLON, SEMICOLON),
                    production_rule!(HEIGHT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(JUSTIFY_CONTENT, COLON, NT_JUSTIFY_CONTENT_TYPE, SEMICOLON),
                    production_rule!(JUSTIFY_ITEMS, COLON, NT_JUSTIFY_ITEMS_TYPE, SEMICOLON),
                    production_rule!(JUSTIFY_SELF, COLON, NT_JUSTIFY_SELF_TYPE, SEMICOLON),
                    production_rule!(KEYFRAMES, COLON, SEMICOLON),
                    production_rule!(LEFT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(LETTER_SPACING, COLON, SEMICOLON),
                    production_rule!(LINE_HEIGHT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(LIST_STYLE, COLON, SEMICOLON),
                    production_rule!(LIST_STYLE_IMAGE, COLON, SEMICOLON),
                    production_rule!(LIST_STYLE_POSITION, COLON, SEMICOLON),
                    production_rule!(LIST_STYLE_TYPE, COLON, SEMICOLON),
                    production_rule!(MARGIN, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(MARGIN_BOTTOM, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(MARGIN_LEFT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(MARGIN_RIGHT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(MARGIN_TOP, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(MAX_HEIGHT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(MAX_WIDTH, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(MEDIA, COLON, SEMICOLON),
                    production_rule!(MIN_HEIGHT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(MIN_WIDTH, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(NAV_DOWN, COLON, SEMICOLON),
                    production_rule!(NAV_INDEX, COLON, SEMICOLON),
                    production_rule!(NAV_LEFT, COLON, SEMICOLON),
                    production_rule!(NAV_RIGHT, COLON, SEMICOLON),
                    production_rule!(NAV_UP, COLON, SEMICOLON),
                    production_rule!(OPACITY, COLON, SEMICOLON),
                    production_rule!(ORDER, COLON, SEMICOLON),
                    production_rule!(OUTLINE, COLON, SEMICOLON),
                    production_rule!(OUTLINE_COLOR, COLON, NT_COLOR, SEMICOLON),
                    production_rule!(OUTLINE_OFFSET, COLON, SEMICOLON),
                    production_rule!(OUTLINE_STYLE, COLON, NT_BORDER_TYPE, SEMICOLON),
                    production_rule!(OUTLINE_WIDTH, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(OVERFLOW, COLON, NT_OVERFLOW_TYPE, SEMICOLON),
                    production_rule!(OVERFLOW_X, COLON, NT_OVERFLOW_TYPE, SEMICOLON),
                    production_rule!(OVERFLOW_Y, COLON, NT_OVERFLOW_TYPE, SEMICOLON),
                    production_rule!(PADDING, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(PADDING_BOTTOM, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(PADDING_LEFT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(PADDING_RIGHT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(PADDING_TOP, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(PAGE_BREAK_AFTER, COLON, SEMICOLON),
                    production_rule!(PAGE_BREAK_BEFORE, COLON, SEMICOLON),
                    production_rule!(PAGE_BREAK_INSIDE, COLON, SEMICOLON),
                    production_rule!(PERSPECTIVE, COLON, SEMICOLON),
                    production_rule!(PERSPECTIVE_ORIGIN, COLON, SEMICOLON),
                    production_rule!(POSITION, COLON, SEMICOLON),
                    production_rule!(QUOTES, COLON, SEMICOLON),
                    production_rule!(RESIZE, COLON, SEMICOLON),
                    production_rule!(RIGHT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_BEHAVIOR, COLON, SEMICOLON),
                    production_rule!(SCROLL_MARGIN, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_MARGIN_BLOCK, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_MARGIN_BLOCK_END, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_MARGIN_BLOCK_START, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_MARGIN_BOTTOM, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_MARGIN_INLINE, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_MARGIN_INLINE_END, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_MARGIN_INLINE_START, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_MARGIN_LEFT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_MARGIN_RIGHT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_MARGIN_TOP, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_PADDING, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_PADDING_BLOCK, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_PADDING_BLOCK_END, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_PADDING_BLOCK_START, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_PADDING_BOTTOM, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_PADDING_INLINE, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_PADDING_INLINE_END, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_PADDING_INLINE_START, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_PADDING_LEFT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_PADDING_RIGHT, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(SCROLL_PADDING_TOP, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(TAB_SIZE, COLON, NT_NUMBER, SEMICOLON),
                    production_rule!(TABLE_LAYOUT, COLON, SEMICOLON),
                    production_rule!(TEXT_ALIGN, COLON, NT_TEXT_ALIGN_TYPE, SEMICOLON),
                    production_rule!(TEXT_ALIGN_LAST, COLON, SEMICOLON),
                    production_rule!(TEXT_DECORATION, COLON, SEMICOLON),
                    production_rule!(TEXT_INDENT, COLON, SEMICOLON),
                    production_rule!(TEXT_OVERFLOW, COLON, SEMICOLON),
                    production_rule!(TEXT_SHADOW, COLON, SEMICOLON),
                    production_rule!(TEXT_TRANSFORM, COLON, SEMICOLON),
                    production_rule!(TOP, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(TRANSFORM, COLON, SEMICOLON),
                    production_rule!(TRANSFORM_ORIGIN, COLON, SEMICOLON),
                    production_rule!(TRANSFORM_STYLE, COLON, SEMICOLON),
                    production_rule!(TRANSITION, COLON, SEMICOLON),
                    production_rule!(TRANSITION_DELAY, COLON, NT_NUMBER, T_LO_S, SEMICOLON),
                    production_rule!(TRANSITION_DURATION, COLON, NT_NUMBER, T_LO_S, SEMICOLON),
                    production_rule!(TRANSITION_PROPERTY, COLON, SEMICOLON),
                    production_rule!(TRANSITION_TIMING_FUNCTION, COLON, SEMICOLON),
                    production_rule!(UNICODE_BIDI, COLON, SEMICOLON),
                    production_rule!(USER_SELECT, COLON, SEMICOLON),
                    production_rule!(VERTICAL_ALIGN, COLON, SEMICOLON),
                    production_rule!(VISIBILITY, COLON, SEMICOLON),
                    production_rule!(WIDOWS, COLON, NT_NUMBER, SEMICOLON),
                    production_rule!(WHITE_SPACE, COLON, SEMICOLON),
                    production_rule!(WIDTH, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(WORD_BREAK, COLON, SEMICOLON),
                    production_rule!(WORD_SPACING, COLON, NT_SIZE, SEMICOLON),
                    production_rule!(WORD_WRAP, COLON, SEMICOLON),
                    production_rule!(Z_INDEX, COLON, NT_NUMBER, SEMICOLON)
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_PSEUDO_CLASSES),
                    vec![
                        production_rule!(LINK),
                        production_rule!(VISITED),
                        production_rule!(HOVER),
                        production_rule!(ACTIVE),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_PSEUDO_ELEMENTS),
                    vec![
                        production_rule!(FIRST_LETTER),
                        production_rule!(FIRST_LINE),
                        production_rule!(MARKER),
                        production_rule!(BEFORE),
                        production_rule!(AFTER),
                        production_rule!(SELECTION),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_DISPLAY_TYPE),
                    vec![
                        production_rule!(INLINE),
                        production_rule!(BLOCK),
                        production_rule!(CONTENTS),
                        production_rule!(FLEX),
                        production_rule!(GRID),
                        production_rule!(INLINE_BLOCK),
                        production_rule!(INLINE_FLEX),
                        production_rule!(INLINE_GRID),
                        production_rule!(INLINE_TABLE),
                        production_rule!(LIST_ITEM),
                        production_rule!(RUN_IN),
                        production_rule!(TABLE),
                        production_rule!(TABLE_CAPTION),
                        production_rule!(TABLE_COLUMN_GROUP),
                        production_rule!(TABLE_HEADER_GROUP),
                        production_rule!(TABLE_FOOTER_GROUP),
                        production_rule!(TABLE_ROW_GROUP),
                        production_rule!(TABLE_CELL),
                        production_rule!(TABLE_COLUMN),
                        production_rule!(TABLE_ROW),
                        production_rule!(NONE),
                        production_rule!(INITIAL),
                        production_rule!(INHERIT),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_ALIGN_CONTENT_TYPE),
                    vec![
                        production_rule!(STRETCH),
                        production_rule!(CENTER),
                        production_rule!(FLEX_START),
                        production_rule!(FLEX_END),
                        production_rule!(SPACE_BETWEEN),
                        production_rule!(SPACE_AROUND),
                        production_rule!(SPACE_EVENLY),
                        production_rule!(INHERIT),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_ALIGN_ITEMS_TYPE),
                    vec![
                        production_rule!(NORMAL),
                        production_rule!(STRETCH),
                        production_rule!(CENTER),
                        production_rule!(FLEX_START),
                        production_rule!(FLEX_END),
                        production_rule!(START),
                        production_rule!(END),
                        production_rule!(BASELINE),
                        production_rule!(INITIAL),
                        production_rule!(INHERIT),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_ALIGN_SELF_TYPE),
                    vec![
                        production_rule!(AUTO),
                        production_rule!(STRETCH),
                        production_rule!(CENTER),
                        production_rule!(FLEX_START),
                        production_rule!(FLEX_END),
                        production_rule!(BASELINE),
                        production_rule!(INITIAL),
                        production_rule!(INHERIT),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_FLOAT_TYPE),
                    vec![
                        production_rule!(NONE),
                        production_rule!(LEFT),
                        production_rule!(RIGHT),
                        production_rule!(INITIAL),
                        production_rule!(INHERIT),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_JUSTIFY_CONTENT_TYPE),
                    vec![
                        production_rule!(FLEX_START),
                        production_rule!(FLEX_END),
                        production_rule!(CENTER),
                        production_rule!(SPACE_BETWEEN),
                        production_rule!(SPACE_AROUND),
                        production_rule!(SPACE_EVENLY),
                        production_rule!(INITIAL),
                        production_rule!(INHERIT),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_JUSTIFY_ITEMS_TYPE),
                    vec![
                        production_rule!(NORMAL),
                        production_rule!(STRETCH),
                        production_rule!(START),
                        production_rule!(LEFT),
                        production_rule!(CENTER),
                        production_rule!(END),
                        production_rule!(RIGHT),
                        production_rule!(INITIAL),
                        production_rule!(INHERIT),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_JUSTIFY_SELF_TYPE),
                    vec![
                        production_rule!(AUTO),
                        production_rule!(NORMAL),
                        production_rule!(STRETCH),
                        production_rule!(START),
                        production_rule!(LEFT),
                        production_rule!(CENTER),
                        production_rule!(END),
                        production_rule!(RIGHT),
                        production_rule!(INITIAL),
                        production_rule!(INHERIT),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_BORDER_TYPE),
                    vec![
                        production_rule!(NONE),
                        production_rule!(HIDDEN),
                        production_rule!(DOTTED),
                        production_rule!(DASHED),
                        production_rule!(SOLID),
                        production_rule!(DOUBLE),
                        production_rule!(GROOVE),
                        production_rule!(RIDGE),
                        production_rule!(INSET),
                        production_rule!(OUTSET),
                        production_rule!(INITIAL),
                        production_rule!(INHERIT),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_OVERFLOW_TYPE),
                    vec![
                        production_rule!(VISIBLE),
                        production_rule!(HIDDEN),
                        production_rule!(CLIP),
                        production_rule!(SCROLL),
                        production_rule!(AUTO),
                        production_rule!(INITIAL),
                        production_rule!(INHERIT),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_TEXT_ALIGN_TYPE),
                    vec![
                        production_rule!(LEFT),
                        production_rule!(RIGHT),
                        production_rule!(JUSTIFY),
                        production_rule!(INITIAL),
                        production_rule!(INHERIT),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_HTML_ELEMENT),
                    vec![
                        production_rule!(NT_BASE_HTML_ELEMENT, COLON, COLON, NT_PSEUDO_ELEMENTS),
                        production_rule!(
                            NT_BASE_HTML_ELEMENT,
                            COLON,
                            COLON,
                            NT_PSEUDO_ELEMENTS,
                            COLON,
                            NT_PSEUDO_CLASSES
                        ),
                        production_rule!(NT_BASE_HTML_ELEMENT, COLON, NT_PSEUDO_CLASSES),
                        production_rule!(NT_BASE_HTML_ELEMENT),
                    ],
                ),
                // Classes variable rule
                Production::new(ProductionLHS::new_context_free_elem(NT_CLASS), classes),
                // Ids variable rule
                Production::new(ProductionLHS::new_context_free_elem(NT_ID), ids),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_BASE_HTML_ELEMENT),
                    vec![
                        production_rule!(A),
                        production_rule!(ABBR),
                        production_rule!(ADDRESS),
                        production_rule!(AREA),
                        production_rule!(ARTICLE),
                        production_rule!(ASIDE),
                        production_rule!(AUDIO),
                        production_rule!(B),
                        production_rule!(BASE),
                        production_rule!(BDI),
                        production_rule!(BDO),
                        production_rule!(BLOCKQUOTE),
                        production_rule!(BODY),
                        production_rule!(BR),
                        production_rule!(BUTTON),
                        production_rule!(CANVAS),
                        production_rule!(CAPTION),
                        production_rule!(CITE),
                        production_rule!(CODE),
                        production_rule!(COL),
                        production_rule!(COLGROUP),
                        production_rule!(DATA),
                        production_rule!(DATALIST),
                        production_rule!(DD),
                        production_rule!(DEL),
                        production_rule!(DETAILS),
                        production_rule!(DFN),
                        production_rule!(DIALOG),
                        production_rule!(DIV),
                        production_rule!(DL),
                        production_rule!(DT),
                        production_rule!(EM),
                        production_rule!(EMBED),
                        production_rule!(FIELDSET),
                        production_rule!(FIGCAPTION),
                        production_rule!(FIGURE),
                        production_rule!(FOOTER),
                        production_rule!(FORM),
                        production_rule!(H1),
                        production_rule!(H2),
                        production_rule!(H3),
                        production_rule!(H4),
                        production_rule!(H5),
                        production_rule!(H6),
                        production_rule!(HEAD),
                        production_rule!(HEADER),
                        production_rule!(HGROUP),
                        production_rule!(HR),
                        production_rule!(HTML),
                        production_rule!(I),
                        production_rule!(IFRAME),
                        production_rule!(IMG),
                        production_rule!(INPUT),
                        production_rule!(INS),
                        production_rule!(KBD),
                        production_rule!(LABEL),
                        production_rule!(LEGEND),
                        production_rule!(LI),
                        production_rule!(LINK),
                        production_rule!(MAIN),
                        production_rule!(MAP),
                        production_rule!(MARK),
                        production_rule!(META),
                        production_rule!(METER),
                        production_rule!(NAV),
                        production_rule!(NOSCRIPT),
                        production_rule!(OBJECT),
                        production_rule!(OL),
                        production_rule!(OPTGROUP),
                        production_rule!(OPTION),
                        production_rule!(OUTPUT),
                        production_rule!(P),
                        production_rule!(PARAM),
                        production_rule!(PICTURE),
                        production_rule!(PRE),
                        production_rule!(PROGRESS),
                        production_rule!(Q),
                        production_rule!(RP),
                        production_rule!(RT),
                        production_rule!(RUBY),
                        production_rule!(S),
                        production_rule!(SAMP),
                        production_rule!(SCRIPT),
                        production_rule!(SECTION),
                        production_rule!(SELECT),
                        production_rule!(SMALL),
                        production_rule!(SOURCE),
                        production_rule!(SPAN),
                        production_rule!(STRONG),
                        production_rule!(STYLE),
                        production_rule!(SUB),
                        production_rule!(SUMMARY),
                        production_rule!(SUP),
                        production_rule!(TABLE),
                        production_rule!(TBODY),
                        production_rule!(TD),
                        production_rule!(TEMPLATE),
                        production_rule!(TEXTAREA),
                        production_rule!(TFOOT),
                        production_rule!(TH),
                        production_rule!(THEAD),
                        production_rule!(TIME),
                        production_rule!(TITLE),
                        production_rule!(TR),
                        production_rule!(TRACK),
                        production_rule!(U),
                        production_rule!(UL),
                        production_rule!(VAR),
                        production_rule!(VIDEO),
                        production_rule!(WBR),
                    ],
                ),
                numeric_character_production_context_free("number".into()),
                Production::new(ProductionLHS::new_context_free_elem(NT_COLOR), colors),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_SIZE),
                    vec![
                        production_rule!(NT_NUMBER, PERCENT),
                        production_rule!(NT_NUMBER, NT_NUMBER, PERCENT),
                        production_rule!(NT_NUMBER, NT_SIZE_SUFFIX),
                        production_rule!(NT_NUMBER, NT_NUMBER, NT_SIZE_SUFFIX),
                        production_rule!(NT_NUMBER, NT_NUMBER, NT_NUMBER, NT_SIZE_SUFFIX),
                    ],
                ),
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_SIZE_SUFFIX),
                    vec![
                        production_rule!(T_LO_C, T_LO_M),                 // cm
                        production_rule!(T_LO_M, T_LO_M),                 // mm
                        production_rule!(T_LO_I, T_LO_N),                 // in
                        production_rule!(T_LO_P, T_LO_X),                 // px
                        production_rule!(T_LO_P, T_LO_T),                 // pt
                        production_rule!(T_LO_P, T_LO_C),                 // pc
                        production_rule!(T_LO_E, T_LO_M),                 // em
                        production_rule!(T_LO_E, T_LO_X),                 // ex
                        production_rule!(T_LO_C, T_LO_H),                 // ch
                        production_rule!(T_LO_R, T_LO_E, T_LO_M),         // rem
                        production_rule!(T_LO_V, T_LO_W),                 // vw
                        production_rule!(T_LO_V, T_LO_H),                 // vh
                        production_rule!(T_LO_V, T_LO_M, T_LO_I, T_LO_N), // vmin
                        production_rule!(T_LO_V, T_LO_M, T_LO_A, T_LO_X), // vmax
                    ],
                ),
            ],
            "css".into(),
        );

        Ok(grammar)
    }
}
