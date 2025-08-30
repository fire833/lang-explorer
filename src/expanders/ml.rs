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
    tensor::{activation::log_softmax, backend::AutodiffBackend, Float, Int, Tensor},
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
                lin3::{Linear3Deep, Linear3DeepConfig},
                lin4::{Linear4Deep, Linear4DeepConfig},
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

    /// When generating outputs, store output tensors here for backpropagation later.
    prod_output_map: HashMap<Production<T, I>, Vec<Tensor<B, 1, Int>>>,

    /// Number of iterations to run to extract words.
    wl_kernel_iterations: u32,
    /// The order in which to hash items when computing new labels.
    wl_kernel_order: WLKernelHashingOrder,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductionModelType {
    Linear2,
    Linear3,
    Linear4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionConfiguration {
    sampling: SamplingStrategy,
    model: ProductionModelType,
    with_bias: bool,
    // Temperature value that will be divided by 1000 to get actual temperature.
    temperature: u32,
}

impl ProductionConfiguration {
    pub const fn new() -> Self {
        Self {
            sampling: SamplingStrategy::HighestProb,
            model: ProductionModelType::Linear2,
            with_bias: true,
            temperature: 1000,
        }
    }
}

impl Default for ProductionConfiguration {
    fn default() -> Self {
        Self::new()
    }
}

/// The different strategies for choosing the next expansion rule
/// given the probability distribution from the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingStrategy {
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
            let module = match production.ml_config.model {
                ProductionModelType::Linear2 => ModuleWrapper::Linear2(
                    Linear2DeepConfig::new(production.len())
                        .with_bias(production.ml_config.with_bias)
                        .init::<B>(&device),
                ),
                ProductionModelType::Linear3 => ModuleWrapper::Linear3(
                    Linear3DeepConfig::new(production.len())
                        .with_bias(production.ml_config.with_bias)
                        .init::<B>(&device),
                ),
                ProductionModelType::Linear4 => ModuleWrapper::Linear4(
                    Linear4DeepConfig::new(production.len())
                        .with_bias(production.ml_config.with_bias)
                        .init::<B>(&device),
                ),
            };

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
            prod_output_map: HashMap::new(),
            embedder: EmbedderWrapper::Doc2Vec(d2v),
            wl_kernel_iterations: 5,
            wl_kernel_order: WLKernelHashingOrder::ParentSelfChildrenOrdered,
        })
    }

    fn expand_rule<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        context: &'a ProgramInstance<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        let doc = context.clone();
        let words =
            doc.extract_words_wl_kernel(self.wl_kernel_iterations, self.wl_kernel_order.clone());

        let embedding = match self.embedder.forward(doc, words) {
            Ok(e) => e,
            Err(err) => panic!("{}", err),
        };

        let model = self
            .production_to_model
            .get(production)
            .unwrap_or_else(|| panic!("could not find model for {:?}", production));

        let output = match model {
            ModuleWrapper::Linear2(linear) => {
                linear.forward(embedding.unsqueeze(), Activation::ReLU)
            }
            ModuleWrapper::Linear3(linear) => {
                linear.forward(embedding.unsqueeze(), Activation::ReLU)
            }
            ModuleWrapper::Linear4(linear) => {
                linear.forward(embedding.unsqueeze(), Activation::ReLU)
            }
        };

        // Optionally scale by temperature.
        let output = if production.ml_config.temperature != 1000 {
            output.div_scalar(production.ml_config.temperature as f32 * 0.001)
        } else {
            output
        };

        let output = log_softmax(output, 1);
        let output = output.squeeze::<1>(0);

        // Depending on our strategy, choose the next expansion.
        let loss = match production.ml_config.sampling {
            SamplingStrategy::HighestProb => output.argmax(0),
            SamplingStrategy::LowestProb => output.argmin(0),
        };

        let index: usize = loss
            .to_data()
            .to_vec::<u32>()
            .expect("unable to cast loss to vec")[0] as usize;

        if let Some(v) = self.prod_output_map.get_mut(production) {
            v.push(loss);
        } else {
            self.prod_output_map.insert(production.clone(), vec![loss]);
        }

        production
            .get(index)
            .expect("couldn't find the selected index")
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
    fn cleanup(&mut self) {
        todo!()
    }
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> LearnedExpander<T, I, B> {}
