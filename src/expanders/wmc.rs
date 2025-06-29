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
    expanders::GrammarExpander,
    grammar::{grammar::Grammar, prod::Production, rule::ProductionRule, NonTerminal, Terminal},
};

/// A Weighted Monte Carlo explorer is a slightly less naive expander
/// that selects paths to go down using a weighted sample from the possible
/// expansion paths available at any given step.
pub struct WeightedMonteCarloExpander {}

impl WeightedMonteCarloExpander {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T, I> GrammarExpander<T, I> for WeightedMonteCarloExpander
where
    T: Terminal,
    I: NonTerminal,
{
    fn expand_rule<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
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
        let sample = rand::random::<f64>() % 1.0;

        let mut idx = production.len() - 1;
        let mut cumsum = 0.0;
        for (i, prob) in distribution.iter().enumerate() {
            cumsum += *prob;
            if sample <= cumsum {
                idx = i;
                break;
            }
        }

        if let Some(rule) = production.get(idx as usize) {
            return rule;
        } else {
            panic!(
                "got an index not valid within production rules: {} out of length: {}",
                idx,
                production.len()
            );
        }
    }

    /// We may need to initialize the expander depending on the type of grammar
    /// we are using. For example, with my ML based example, the internal models of
    /// the expander may change completely depending on the rules of the grammar
    /// I want to expand.
    fn init<'a>(_grammar: &'a Grammar<T, I>) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        Ok(Self::new())
    }
}
