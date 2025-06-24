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

use std::fmt::Debug;

pub mod alphanumeric;

use crate::grammar::{elem::GrammarElement, BinarySerialize, NonTerminal, Terminal};

/// A generic terminal or non-terminal type which wraps a standard String.
/// This is generally the base atomic types that are going to be referenced within
/// grammars. The code for grammars is generic for the purpose of conveniently
/// working with binary languages if desired, but usually string grammars will work
/// fine for a majority of people.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct StringValue {
    s: &'static str,

    /// For dynamic values, allow the string to be heap allocated.
    other: Option<String>,
}

/// StringValue is a valid terminal.
impl Terminal for StringValue {}

/// StringValue is a valid non-terminal.
impl NonTerminal for StringValue {}

impl BinarySerialize for StringValue {
    fn serialize(&self) -> Vec<u8> {
        if let Some(other) = &self.other {
            return other.as_bytes().to_vec();
        }
        self.s.as_bytes().to_vec()
    }

    fn serialize_into(&self, output: &mut Vec<u8>) {
        output.append(&mut self.s.as_bytes().to_vec());
    }
}

impl StringValue {
    pub const fn from_static_str(value: &'static str) -> Self {
        Self {
            s: value,
            other: None,
        }
    }
}

impl Debug for StringValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(other) = &self.other {
            write!(f, "{}", other)?;
        }

        write!(f, "{}", self.s)
    }
}

impl From<&'static str> for StringValue {
    fn from(value: &'static str) -> Self {
        Self::from_static_str(value)
    }
}

impl From<&String> for StringValue {
    fn from(value: &String) -> Self {
        Self {
            s: "",
            other: Some(value.clone()),
        }
    }
}

impl From<String> for StringValue {
    fn from(value: String) -> Self {
        Self {
            s: "",
            other: Some(value),
        }
    }
}

/// Macro to take a string literal and convert into a constant StringValue.
/// This is useful for statically defining your terminals for easy reuse.
macro_rules! terminal_str {
    ($v:vis, $i:ident, $s:literal) => {
        $v const $i: GrammarElement<StringValue, StringValue> =
            GrammarElement::Terminal(StringValue::from_static_str($s));
    };
    ($i:ident, $s:literal) => {
        const $i: GrammarElement<StringValue, StringValue> =
            GrammarElement::Terminal(StringValue::from_static_str($s));
    };
}

pub(crate) use terminal_str;

macro_rules! nterminal_str {
    ($v:vis, $i:ident, $s:literal) => {
        $v fn $i: GrammarElement<StringValue, StringValue> =
            GrammarElement::NonTerminal(StringValue::from_static_str($s));
    };
    ($i:ident, $s:literal) => {
        const $i: GrammarElement<StringValue, StringValue> =
            GrammarElement::NonTerminal(StringValue::from_static_str($s));
    }
}

pub(crate) use nterminal_str;

// Some common terminals/tokens that you can import for other grammars.
pub const EPSILON: GrammarElement<StringValue, StringValue> = GrammarElement::Epsilon;

terminal_str!(pub, COMMA, ",");
terminal_str!(pub, DOT, ".");
terminal_str!(pub, NOT, "!");
terminal_str!(pub, OR, "|");
terminal_str!(pub, AND, "&");
terminal_str!(pub, LPAREN, "(");
terminal_str!(pub, RPAREN, ")");
terminal_str!(pub, LBRACE, "[");
terminal_str!(pub, RBRACE, "]");
terminal_str!(pub, LBRACKET, "{");
terminal_str!(pub, RBRACKET, "}");
terminal_str!(pub, SEMICOLON, ";");
terminal_str!(pub, COLON, ":");
terminal_str!(pub, PLUS, "+");
terminal_str!(pub, MINUS, "-");
terminal_str!(pub, FORWARDSLASH, "/");
terminal_str!(pub, BACKSLASH, "\\");
terminal_str!(pub, EQUALS, "=");
terminal_str!(pub, GREATER, ">");
terminal_str!(pub, LESS, "<");
terminal_str!(pub, STAR, "*");
terminal_str!(pub, HASH, "#");
terminal_str!(pub, SPACE, " ");
