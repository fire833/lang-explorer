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

use std::fmt::{Debug, Display};

use crate::grammar::{elem::GrammarElement, NonTerminal, Terminal};

/// A production rule to use for grammar expansion. Contains a list of
/// GrammarElements that are expanded usually using DFS until only a list of
/// non-terminals remains.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ProductionRule<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    pub items: Vec<GrammarElement<T, I>>,
    pub logit: Option<u64>,
}

impl<T, I> ProductionRule<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    pub const fn new(elements: Vec<GrammarElement<T, I>>) -> Self {
        Self {
            items: elements,
            logit: None,
        }
    }

    pub const fn new_with_logit(elements: Vec<GrammarElement<T, I>>, logit: u64) -> Self {
        Self {
            items: elements,
            logit: Some(logit),
        }
    }
}

impl<T, I> Display for ProductionRule<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, item) in self.items.iter().enumerate() {
            if i == self.items.len() - 1 {
                write!(f, "{}", item)?;
            } else {
                write!(f, "{} ", item)?;
            }
        }

        Ok(())
    }
}

impl<T, I> Debug for ProductionRule<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for item in self.items.iter() {
            write!(f, "{:?}", item)?;
        }

        Ok(())
    }
}
