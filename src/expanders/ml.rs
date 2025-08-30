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

use core::panic;
use std::collections::HashMap;

use burn::{
    optim::AdamWConfig,
    prelude::Backend,
    tensor::{activation::log_softmax, backend::AutodiffBackend, Float, Tensor},
};

use crate::{
    embedding::{
        doc2vecdbowns::{Doc2VecDBOWNSEmbedderParams, Doc2VecEmbedderDBOWNS},
        GeneralEmbeddingTrainingParams, LanguageEmbedder,
    },
    errors::LangExplorerError,
    expanders::GrammarExpander,
    grammar::{
        grammar::Grammar,
        lhs::ProductionLHS,
        prod::Production,
        program::{ProgramInstance, WLKernelHashingOrder},
        rule::ProductionRule,
        NonTerminal, Terminal,
    },
    languages::Feature,
    tooling::{
        modules::{
            embed::AggregationMethod,
            expander::{
                lin2::{Linear2Deep, Linear2DeepConfig},
                lin3::Linear3Deep,
                lin4::Linear4Deep,
                Activation,
            },
        },
        training::TrainingParams,
    },
};

pub struct LearnedExpander<T: Terminal, I: NonTerminal, B: AutodiffBackend> {
    /// The embedder to create embeddings for a
    /// new program or partial program.
    embedder: EmbedderWrapper<T, I, B>,

    /// Mapping of each production rule to its corresponding decision function.
    production_to_model: HashMap<Production<T, I>, ModuleWrapper<B>>,

    strategy: SamplingStrategy,
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
enum ModuleWrapper<B: Backend> {
    Linear2(Linear2Deep<B>),
    Linear3(Linear3Deep<B>),
    Linear4(Linear4Deep<B>),
}

/// Another hack to allow us to use multiple embedders.
enum EmbedderWrapper<T: Terminal, I: NonTerminal, B: AutodiffBackend> {
    Doc2Vec(Doc2VecEmbedderDBOWNS<T, I, B>),
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> EmbedderWrapper<T, I, B> {
    fn forward(
        &mut self,
        doc: ProgramInstance<T, I>,
        words: Vec<Feature>,
    ) -> Result<Tensor<B, 1, Float>, LangExplorerError> {
        match self {
            EmbedderWrapper::Doc2Vec(d2ve) => d2ve.embed((doc.to_string(), words)),
        }
    }
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> GrammarExpander<T, I>
    for LearnedExpander<T, I, B>
{
    fn init(grammar: &Grammar<T, I>, _seed: u64) -> Result<Self, LangExplorerError> {
        let device = Default::default();
        let mut map = HashMap::new();

        for production in grammar.get_productions() {
            let module = ModuleWrapper::Linear2(
                Linear2DeepConfig::new(production.len())
                    .with_bias(true)
                    .init(&device),
            );

            map.insert(production.clone(), module);
        }

        let d2v = Doc2VecEmbedderDBOWNS::new(
            grammar,
            Doc2VecDBOWNSEmbedderParams::new(
                AdamWConfig::new(),
                1000,
                1000,
                GeneralEmbeddingTrainingParams::new(
                    AggregationMethod::Average,
                    TrainingParams::new(),
                ),
                "".to_string(),
            ),
            device,
        );

        Ok(Self {
            production_to_model: map,
            strategy: SamplingStrategy::HighestProb,
            embedder: EmbedderWrapper::Doc2Vec(d2v),
        })
    }

    fn expand_rule<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        context: &'a ProgramInstance<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        let doc = context.clone();
        let words = doc.extract_words_wl_kernel(5, WLKernelHashingOrder::ParentSelfChildrenOrdered);

        let embedding = match self.embedder.forward(doc, words) {
            Ok(e) => e,
            Err(err) => panic!("{}", err),
        };

        let model = self
            .production_to_model
            .get(production)
            .unwrap_or_else(|| panic!("could not find model for {:?}", production));
        let distribution = match model {
            ModuleWrapper::Linear2(linear) => {
                log_softmax(linear.forward(embedding.unsqueeze(), Activation::ReLU), 0)
            }
            ModuleWrapper::Linear3(linear) => {
                log_softmax(linear.forward(embedding.unsqueeze(), Activation::ReLU), 0)
            }
            ModuleWrapper::Linear4(linear) => {
                log_softmax(linear.forward(embedding.unsqueeze(), Activation::ReLU), 0)
            }
        }
        .to_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

        // Depending on our strategy, choose the next expansion.
        let index: usize = match self.strategy {
            SamplingStrategy::Random => {
                // Sample in [0, 1].
                let sample = rand::random::<f32>() % 1.0;
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
                let mut lowest = f32::MAX;
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

        production.get(index).unwrap()
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
}
