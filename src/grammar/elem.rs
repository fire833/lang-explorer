/*
*	Copyright (C) 2026 Kendall Tauser
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

use std::fmt::{Debug, Display};

use crate::grammar::{NonTerminal, Terminal};

/// The atomic elements that comprise the grammar. These can be terminals,
/// which should serialize to a set of bytes (i.e. become valid program code)
/// non-terminals, which are used within an AST representation.
#[derive(Clone, Hash, PartialEq, Eq)]
pub enum GrammarElement {
    Terminal(Terminal),
    NonTerminal(NonTerminal),
    Epsilon,
}

impl From<Terminal> for GrammarElement {
    fn from(value: Terminal) -> Self {
        Self::Terminal(value)
    }
}

impl From<NonTerminal> for GrammarElement {
    fn from(value: NonTerminal) -> Self {
        Self::NonTerminal(value)
    }
}

impl Display for GrammarElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Terminal(t) => write!(f, "'{t:?}'"),
            Self::NonTerminal(nt) => write!(f, "<{nt:?}>"),
            Self::Epsilon => write!(f, "'ε'"),
        }
    }
}

impl Debug for GrammarElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Epsilon => write!(f, "ε"),
            Self::NonTerminal(nt) => write!(f, "{nt:?}"),
            Self::Terminal(t) => write!(f, "{t:?}"),
        }
    }
}
