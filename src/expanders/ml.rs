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

use core::f64;
use std::collections::HashMap;

use burn::{
    nn::{Linear, LinearConfig},
    prelude::Backend,
    record::{BinGzFileRecorder, FullPrecisionSettings},
    tensor::{activation::log_softmax, Device, Tensor},
};

use crate::{
    errors::LangExplorerError,
    expanders::GrammarExpander,
    grammar::{grammar::Grammar, prod::Production, rule::ProductionRule, NonTerminal, Terminal},
};

pub struct LearnedExpander<T, I, B>
where
    T: Terminal,
    I: NonTerminal,
    B: Backend,
{
    /// Embedding model that creates embeddings of grammars.
    // embedding: Embedding<B>,

    /// Recorder to store/load models from disk.
    _recorder: BinGzFileRecorder<FullPrecisionSettings>,

    /// The device on which things should live.
    dev: Device<B>,

    /// Mapping of each production rule to its corresponding decision function.
    production_to_model: HashMap<Production<T, I>, ModuleWrapper<B>>,

    strategy: SamplingStrategy,
}

impl<T, I, B> LearnedExpander<T, I, B>
where
    T: Terminal,
    I: NonTerminal,
    B: Backend,
{
    pub fn new() -> Self {
        Self {
            _recorder: BinGzFileRecorder::new(),
            dev: Default::default(),
            production_to_model: HashMap::new(),
            strategy: SamplingStrategy::Random,
        }
    }
}

/// The different strategies for choosing the next expansion rule
/// given the probability distribution from the model.
pub enum SamplingStrategy {
    /// Randomly sample from the distribution.
    Random,

    /// Choose the highest probability expansion.
    HighestProb,

    /// Choose the lowest probability distribution,
    /// you probably don't want to do this if you
    /// care about your output.
    LowestProb,
}

/// A bit of a hack to allow us to keep a mapping of models
/// for each of the production rules in our grammar.
enum ModuleWrapper<B>
where
    B: Backend,
{
    Linear(Linear<B>),
    // Conv1d(Conv1d<B>),
    // Conv2d(Conv2d<B>),
}

impl<T, I, B> GrammarExpander<T, I> for LearnedExpander<T, I, B>
where
    T: Terminal,
    I: NonTerminal,
    B: Backend,
{
    fn init<'a>(grammar: &'a Grammar<T, I>) -> Result<Self, LangExplorerError> {
        let device = Default::default();
        let recorder = BinGzFileRecorder::<FullPrecisionSettings>::new();
        let mut map = HashMap::new();

        for production in grammar.get_productions() {
            let module = ModuleWrapper::Linear(
                LinearConfig::new(128, production.len())
                    .with_bias(true)
                    .init(&device),
            );

            map.insert(production.clone(), module);
        }

        Ok(Self {
            production_to_model: map,
            _recorder: recorder,
            strategy: SamplingStrategy::Random,
            // Default this for now.
            dev: device,
        })
    }

    fn expand_rule<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        if let Some(model) = self.production_to_model.get(&production) {
            let distribution = match model {
                ModuleWrapper::Linear(linear) => {
                    log_softmax(linear.forward(Tensor::<B, 1>::ones([128], &self.dev)), 0)
                }
            }
            .to_data()
            .convert::<f64>()
            .to_vec()
            .unwrap();

            // Sample in [0, 1].
            let sample = rand::random::<f64>();

            // Depending on our strategy, choose the next expansion.
            let index: usize = match self.strategy {
                SamplingStrategy::Random => {
                    let mut idx = production.len() - 1;
                    let mut cumsum = 0.0;
                    for (i, prob) in distribution.iter().enumerate() {
                        cumsum += prob;
                        if sample <= cumsum {
                            idx = i;
                            break;
                        }
                    }

                    idx
                }
                SamplingStrategy::HighestProb => {
                    let mut highest = 0.0;
                    let mut highest_idx = 0;

                    for (i, prob) in distribution.iter().enumerate() {
                        if *prob > highest {
                            highest = *prob;
                            highest_idx = i;
                        }
                    }

                    highest_idx
                }
                SamplingStrategy::LowestProb => {
                    let mut lowest = f64::MAX;
                    let mut lowest_idx = 0;

                    for (i, prob) in distribution.iter().enumerate() {
                        if *prob < lowest {
                            lowest = *prob;
                            lowest_idx = i;
                        }
                    }

                    lowest_idx
                }
            };

            return production.get(index).unwrap();
        } else {
            panic!(
                "expander does not have model for production rule {:?}",
                production
            );
        }
    }
}
