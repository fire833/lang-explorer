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

use crate::grammar::BinarySerialize;

pub mod taco_schedule;

/// A generic terminal or non-terminal type which wraps a standard String.
/// This is generally the base atomic types that are going to be referenced within
/// grammars. The code for grammars is generic for the purpose of conveniently
/// working with binary languages if desired, but usually string grammars will work
/// fine for a majority of people.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct StringValue {
    s: String,
}

impl BinarySerialize for StringValue {
    fn serialize(&self) -> Vec<u8> {
        self.s.clone().into_bytes()
    }

    fn serialize_into(&self, output: &mut Vec<u8>) {
        output.append(&mut self.s.clone().into_bytes());
    }
}

impl Debug for StringValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.s)
    }
}

impl From<&str> for StringValue {
    fn from(value: &str) -> Self {
        Self {
            s: String::from(value),
        }
    }
}

impl From<String> for StringValue {
    fn from(value: String) -> Self {
        Self { s: value }
    }
}

/// Macro to take a string literal and convert into a constant StringValue.
/// This is useful for statically defining your terminals for easy reuse.
macro_rules! const_string_value {
    ($s:literal) => {};
}
pub(crate) use const_string_value;
