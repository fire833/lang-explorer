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

use std::{
    fmt::{Debug, Display},
    hash::{DefaultHasher, Hash, Hasher},
};

use crate::{
    grammar::{lhs::ProductionLHS, rule::ProductionRule, NonTerminal, Terminal},
    tooling::modules::expander::ProductionConfiguration,
};

/// Represents all the expansion rules for a particular non-terminal
/// identifier.
#[derive(Clone, PartialEq, Eq)]
pub struct Production<T: Terminal, I: NonTerminal> {
    /// Reference to the non-terminal that we are using here.
    pub non_terminal: ProductionLHS<T, I>,

    /// The list of all production rules (ie vectors of vectors of symbols
    /// that can be expanded upon in the grammar expansion process).
    pub items: Vec<ProductionRule<T, I>>,

    /// Configuration for LearnedExpander to build a model for predicting this
    /// production.
    pub ml_config: ProductionConfiguration,
}

impl<T: Terminal, I: NonTerminal> Production<T, I> {
    pub const fn new(non_terminal: ProductionLHS<T, I>, items: Vec<ProductionRule<T, I>>) -> Self {
        Self {
            items,
            non_terminal,
            ml_config: ProductionConfiguration::new(),
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

    pub fn is_empty(&self) -> bool {
        self.items.len() == 0
    }

    /// Wrapper to return a specific rule.
    pub fn get(&self, i: usize) -> Option<&ProductionRule<T, I>> {
        self.items.get(i)
    }

    pub(crate) fn hash_internal(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

impl<T: Terminal, I: NonTerminal> Display for Production<T, I> {
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

impl<T: Terminal, I: NonTerminal> Debug for Production<T, I> {
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

impl<T: Terminal, I: NonTerminal> Hash for Production<T, I> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.non_terminal.hash(state);
        self.items.hash(state);
    }
}

macro_rules! production_rule {
    ($logit:literal, $($x:expr),+) => {
        // Create a production rule with logits.
        ProductionRule::new_with_logit(vec![$($x),+], $logit as u64)
    };
    ($($x:expr),+) => {
        // Create a production rule withot logits.
        ProductionRule::new(vec![$($x),+])
    }
}

pub(crate) use production_rule;

macro_rules! context_free_production {
    ($nt:expr, $($rules:expr),+) => {
        Production::new(ProductionLHS::new_context_free_elem($nt), vec![$($rules),+])
    };
    // ($nt:expr, $rules:expr) => {
    //     Production::new(
    //         ProductionLHS::new_context_free_elem($nt),
    //         $rules,
    //     )
    // }
}

pub(crate) use context_free_production;

macro_rules! single_prefix_production {
    ($prefix: expr, $nt: expr, $($rules:expr),+) => {
        Production::new(ProductionLHS::new_with_prefix_single($prefix, $nt), vec![$($rules),+])
    };
}

macro_rules! single_suffix_production {
    ($nt: expr, $suffix: expr, $($rules:expr),+) => {
        Production::new(ProductionLHS::new_with_suffix_single($suffix, $nt), vec![$($rules),+])
    };
}

pub(crate) use single_prefix_production;
pub(crate) use single_suffix_production;

#[allow(unused)]
macro_rules! context_sensitive_production {
    (($($prefix:expr),*) / ($nt:expr) / ($($suffix:expr),*) / ($($rules:expr),+)) => {
        Production::new(ProductionLHS::new_with_prefix_and_suffix(vec![$($prefix),+], $nt, vec![$($suffix),+]), vec![$($rules),+])
    };
}

#[allow(unused)]
pub(crate) use context_sensitive_production;
