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
    expanders::GrammarExpander,
    grammar::{
        grammar::Grammar, lhs::ProductionLHS, prod::Production, program::ProgramInstance,
        rule::ProductionRule, NonTerminal, Terminal,
    },
};

/// A Weighted Monte Carlo explorer is a slightly less naive expander
/// that selects paths to go down using a weighted sample from the possible
/// expansion paths available at any given step.
pub struct WeightedMonteCarloExpander {
    rng: ChaCha8Rng,
}

impl WeightedMonteCarloExpander {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl<T: Terminal, I: NonTerminal> GrammarExpander<T, I> for WeightedMonteCarloExpander {
    /// We may need to initialize the expander depending on the type of grammar
    /// we are using. For example, with my ML based example, the internal models of
    /// the expander may change completely depending on the rules of the grammar
    /// I want to expand.
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
        let mut sum = 0;
        let mut count = 0;

        let mut logits: Vec<u64> = vec![];

        for rule in production.items.iter() {
            if let Some(logit) = rule.logit {
                if logit == 0 {
                    // Make sure we can't have zero for a logit.
                    sum += 1;
                } else {
                    sum += logit;
                }
                logits.push(logit);
                count += 1;
            } else {
                logits.push(0);
            }
        }

        if count == 0 {
            count = 1;
        }

        let mut avg = sum / count;
        if avg == 0 {
            avg = 1;
        }

        let mut total: f64 = 0.0;

        for (i, _) in production.items.iter().enumerate() {
            if logits[i] == 0 {
                logits[i] = avg; // Make sure we can't have zero for a logit.
            }

            total += logits[i] as f64;
        }

        // Take the softmax
        let distribution: Vec<f64> = logits.iter().map(|item| *item as f64 / total).collect();
        let sample = self.rng.random::<f64>() % 1.0;

        let mut idx = production.len() - 1;
        let mut cumsum = 0.0;
        for (i, prob) in distribution.iter().enumerate() {
            cumsum += *prob;
            if sample <= cumsum {
                idx = i;
                break;
            }
        }

        if let Some(rule) = production.get(idx) {
            rule
        } else {
            panic!(
                "got an index not valid within production rules: {} out of length: {}",
                idx,
                production.len()
            );
        }
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
        _lhs_location_matrix: &[(&'a ProductionLHS<T, I>, Vec<usize>)],
    ) -> (&'a ProductionLHS<T, I>, usize) {
        todo!()
    }

    /// Whenever a program has finished being generated, this method will be called
    /// to reset/update internal state in the expander. This is mostly going to be used
    /// in the learned expander to run backprop and update the internal models for
    /// generating the program in the first place.
    fn cleanup(&mut self) {}
}
