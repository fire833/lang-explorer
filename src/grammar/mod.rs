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

pub mod elem;
pub mod grammar;
pub mod lhs;
pub mod prod;
pub mod program;
pub mod rule;

use std::fmt::{Debug, Display};

use crate::grammar::elem::GrammarElement;

/// Trait for non-terminals to implement in order to be serialized
/// to an output program.
pub trait BinarySerialize {
    /// Serialize into a Vec.
    fn serialize(&self) -> Vec<u8>;

    /// Serialize by appending to the output vector.
    fn serialize_into(&self, output: &mut Vec<u8>);
}

/// Wrapper for all terminal elements.
#[derive(Clone, Hash, PartialEq, Eq)]
pub enum Terminal {
    String(String),
    ConstStr(&'static str),
    Byte(u8),
}

impl Display for Terminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "\"{s}\""),
            Self::ConstStr(s) => write!(f, "\"{s}\""),
            Self::Byte(b) => write!(f, "{b:#04x}"),
        }
    }
}

impl Debug for Terminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "{s}"),
            Self::ConstStr(s) => write!(f, "{s}"),
            Self::Byte(b) => write!(f, "{b:#04x}"),
        }
    }
}

impl From<&'static str> for Terminal {
    fn from(value: &'static str) -> Self {
        Self::ConstStr(value)
    }
}

impl From<String> for Terminal {
    fn from(value: String) -> Self {
        Self::String(value.clone())
    }
}

impl<'a> From<&'a String> for Terminal {
    fn from(value: &'a String) -> Self {
        Self::String(value.clone())
    }
}

impl From<u8> for Terminal {
    fn from(value: u8) -> Self {
        Self::Byte(value)
    }
}

impl BinarySerialize for Terminal {
    fn serialize(&self) -> Vec<u8> {
        match self {
            Terminal::String(s) => s.as_bytes().to_vec(),
            Terminal::ConstStr(s) => s.as_bytes().to_vec(),
            Terminal::Byte(b) => vec![*b],
        }
    }

    fn serialize_into(&self, output: &mut Vec<u8>) {
        output.append(&mut self.serialize());
    }
}

/// Wrapper for all non-terminal elements.
#[derive(Clone, Hash, PartialEq, Eq)]
pub enum NonTerminal {
    ConstStr(&'static str),
}

impl Display for NonTerminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstStr(s) => write!(f, "{s}"),
        }
    }
}

impl Debug for NonTerminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstStr(s) => write!(f, "{s}"),
        }
    }
}

impl From<&'static str> for NonTerminal {
    fn from(value: &'static str) -> Self {
        Self::ConstStr(value)
    }
}
