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

/// Represents all the expansion rules for a particular non-terminal
/// identifier.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Production<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    /// Reference to the non-terminal that we are using here.
    pub non_terminal: ProductionLHS<T, I>,

    /// The list of all production rules (ie vectors of vectors of symbols
    /// that can be expanded upon in the grammar expansion process).
    pub items: Vec<ProductionRule<T, I>>,
}

macro_rules! production_rule {
    ($($x:expr),+) => {
        ProductionRule::new(vec![$($x),+])
    };
}

use std::fmt::{Debug, Display};

use burn::{module::Module, nn, prelude::Backend};
pub(crate) use production_rule;

macro_rules! context_free_production {
    ($nt:expr, $($x:expr),+) => {
        Production::new(
            ProductionLHS::new_context_free_elem($nt), vec![$($x),+],
        )
    };
    // ($nt:expr, $rules:expr) => {
    //     Production::new(
    //         ProductionLHS::new_context_free_elem($nt),
    //         $rules,
    //     )
    // }
}

pub(crate) use context_free_production;

use crate::grammar::{lhs::ProductionLHS, rule::ProductionRule, NonTerminal, Terminal};

impl<T, I> Production<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    pub const fn new(non_terminal: ProductionLHS<T, I>, items: Vec<ProductionRule<T, I>>) -> Self {
        Self {
            items,
            non_terminal,
        }
    }

    /// Get the left-hand size value for this production.
    pub fn lhs(&self) -> ProductionLHS<T, I> {
        self.non_terminal.clone()
    }

    /// Wrapper to return an iterator for all production rules in this production.
    pub fn iter(&self) -> impl Iterator<Item = &ProductionRule<T, I>> + '_ {
        self.items.iter()
    }

    /// Wrapper to return number of production rules in this production.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Wrapper to return a specific rule.
    pub fn get(&self, i: usize) -> Option<&ProductionRule<T, I>> {
        self.items.get(i)
    }

    pub fn create_linear_classifier<B: Backend>(
        &self,
        embedding_dim: u32,
        device: &B::Device,
    ) -> impl Module<B> {
        nn::LinearConfig::new(embedding_dim as usize, self.items.len())
            .with_bias(true)
            .init(device)
    }
}

impl<T, I> Display for Production<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, item) in self.items.iter().enumerate() {
            if i == self.items.len() - 1 {
                write!(f, " {}", item)?;
            } else if i == 0 {
                write!(f, "{} |", item)?;
            } else {
                write!(f, " {} |", item)?;
            }
        }

        Ok(())
    }
}

impl<T, I> Debug for Production<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, item) in self.items.iter().enumerate() {
            if i == self.items.len() - 1 {
                write!(f, " {:?}", item)?;
            } else if i == 0 {
                write!(f, "{:?} |", item)?;
            } else {
                write!(f, " {:?} |", item)?;
            }
        }

        Ok(())
    }
}
