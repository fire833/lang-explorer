/*
*	Copyright (C) 2024 Kendall Tauser
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

use crate::grammar::{BinarySerialize, GrammarElement, NonTerminal, Terminal};

pub mod taco_schedule;

/// A generic terminal or non-terminal type which wraps a standard String.
/// This is generally the base atomic types that are going to be referenced within
/// grammars. The code for grammars is generic for the purpose of conveniently
/// working with binary languages if desired, but usually string grammars will work
/// fine for a majority of people.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct StringValue {
    s: &'static str,
}

/// StringValue is a valid terminal.
impl Terminal for StringValue {}

/// StringValue is a valid non-terminal.
impl NonTerminal for StringValue {}

impl BinarySerialize for StringValue {
    fn serialize(&self) -> Vec<u8> {
        self.s.as_bytes().to_vec()
    }

    fn serialize_into(&self, output: &mut Vec<u8>) {
        output.append(&mut self.s.as_bytes().to_vec());
    }
}

impl StringValue {
    const fn from_static_str(value: &'static str) -> Self {
        Self { s: value }
    }
}

impl Debug for StringValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.s)
    }
}

impl From<&'static str> for StringValue {
    fn from(value: &'static str) -> Self {
        Self::from_static_str(value)
    }
}

/// Macro to take a string literal and convert into a constant StringValue.
/// This is useful for statically defining your terminals for easy reuse.
macro_rules! terminal_str {
    ($i:ident, $s:literal) => {
        const $i: GrammarElement<StringValue, StringValue> =
            GrammarElement::Terminal(StringValue::from_static_str($s));
    };
}
pub(crate) use terminal_str;

macro_rules! nterminal_str {
    ($i:ident, $s:literal) => {
        const $i: GrammarElement<StringValue, StringValue> =
            GrammarElement::NonTerminal(StringValue::from_static_str($s));
    };
}

pub(crate) use nterminal_str;

// Some common terminals/tokens that you can import for other grammars.
