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

use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

use crate::{
    errors::LangExplorerError,
    grammar::{
        grammar::Grammar, lhs::ProductionLHS, prod::Production, program::ProgramInstance,
        rule::ProductionRule, NonTerminal, Terminal,
    },
};

use super::GrammarExpander;

/// A Monte-Carlo expander is a naive expander that randomly
/// selects paths to go down within the range of possible outcomes.
/// This could lead to very dumb outputs, and take a very long time to
/// create fully terminated words in a particular language.
pub struct MonteCarloExpander {
    rng: ChaCha8Rng,
}

impl MonteCarloExpander {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl<T: Terminal, I: NonTerminal> GrammarExpander<T, I> for MonteCarloExpander {
    fn init(_grammar: &Grammar<T, I>, seed: u64) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        Ok(Self::new(seed))
    }

    fn expand_rule<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        _context: &'a ProgramInstance<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        let idx = self.rng.random::<u64>() as usize % production.len();
        production.get(idx).expect("got out of bounds index")
    }

    /// For context sensitive grammars, we could be in a situation where we have
    /// multiple left-hand sides that match some point on the frontier, along with
    /// multiple positions within the frontier where we could expand such left-hand side
    /// with a production. Thus, we want the expander to have the ability to make this
    /// decision on our behalf as well.
    fn choose_lhs_and_slot<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        _context: &'a ProgramInstance<T, I>,
        lhs_location_matrix: &[(&'a ProductionLHS<T, I>, Vec<usize>)],
    ) -> (&'a ProductionLHS<T, I>, usize) {
        let idx = self.rng.random::<u64>() as usize % lhs_location_matrix.len();
        let (lhs, indices) = lhs_location_matrix
            .get(idx)
            .expect("got out of bounds index for lhs");
        let idx = self.rng.random::<u64>() as usize % indices.len();
        let index = indices
            .get(idx)
            .expect("got invalid index for frontier indices");
        (*lhs, *index)
    }

    /// Whenever a program has finished being generated, this method will be called
    /// to reset/update internal state in the expander. This is mostly going to be used
    /// in the learned expander to run backprop and update the internal models for
    /// generating the program in the first place.
    fn cleanup(&mut self) {}
}
