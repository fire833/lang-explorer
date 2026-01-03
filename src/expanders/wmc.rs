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

use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

use crate::{
    errors::LangExplorerError,
    expanders::GrammarExpander,
    grammar::{
        grammar::Grammar, lhs::ProductionLHS, prod::Production, program::ProgramInstance,
        rule::ProductionRule,
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

impl GrammarExpander for WeightedMonteCarloExpander {
    fn init(_grammar: &Grammar, seed: u64) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        Ok(Self::new(seed))
    }

    fn expand_rule<'a>(
        &mut self,
        _grammar: &'a Grammar,
        _context: &'a ProgramInstance,
        production: &'a Production,
    ) -> &'a ProductionRule {
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

    fn choose_lhs_and_slot<'a>(
        &mut self,
        _grammar: &'a Grammar,
        _context: &'a ProgramInstance,
        lhs_location_matrix: &[(&'a ProductionLHS, Vec<usize>)],
    ) -> (&'a ProductionLHS, usize) {
        // For now, copy mc
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

    fn cleanup(&mut self) {}
}
