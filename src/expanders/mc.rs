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

use rand::{rngs::ThreadRng, Rng};

use crate::{
    errors::LangExplorerError,
    grammar::{prod::Production, rule::ProductionRule, Grammar, NonTerminal, Terminal},
};

use super::GrammarExpander;

/// A Monte-Carlo expander is a naive expander that randomly
/// selects paths to go down within the range of possible outcomes.
/// This could lead to very dumb outputs, and take a very long time to
/// create fully terminated words in a particular language.
pub struct MonteCarloExpander {
    rng: ThreadRng,
}

impl MonteCarloExpander {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }
}

impl<T, I> GrammarExpander<T, I> for MonteCarloExpander
where
    T: Terminal,
    I: NonTerminal,
{
    fn init<'a>(_grammar: &'a Grammar<T, I>) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        Ok(Self::new())
    }

    fn expand_rule<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        let len = production.len();
        let val = self.rng.gen::<u64>() % len as u64;
        if let Some(rule) = production.get(val as usize) {
            return rule;
        } else {
            panic!(
                "got an index not valid within production rules: {} out of length: {}",
                val, len
            );
        }
    }
}
