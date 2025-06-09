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

use std::collections::HashMap;

use burn::{
    nn::{Linear, LinearConfig},
    prelude::Backend,
    record::{BinGzFileRecorder, FullPrecisionSettings},
    tensor::{activation::softmax, Device, Tensor},
};

use crate::{
    errors::LangExplorerError,
    expanders::GrammarExpander,
    grammar::{Grammar, NonTerminal, Production, ProductionRule, Terminal},
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
    recorder: BinGzFileRecorder<FullPrecisionSettings>,

    /// The device on which things should live.
    dev: Device<B>,

    /// Mapping of each production rule to its corresponding decision function.
    production_to_model: HashMap<Production<T, I>, ModuleWrapper<B>>,
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
            recorder,
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
                    softmax(linear.forward(Tensor::ones([128], &self.dev)), 0)
                }
            };

            // Sample in [0, 1].
            let sample = rand::random::<f64>() / f64::MAX;

            for item in distribution.iter_dim(0 as usize) {}

            return production.get(0).unwrap();
        } else {
            panic!(
                "expander does not have model for production rule {:?}",
                production
            );
        }
    }
}
