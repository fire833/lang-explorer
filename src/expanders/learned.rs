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

use burn::{optim::AdamWConfig, tensor::backend::AutodiffBackend};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    embedding::{
        doc2vecdbowns::{Doc2VecDBOWNSEmbedderParams, Doc2VecEmbedderDBOWNS},
        GeneralEmbeddingTrainingParams, LanguageEmbedder,
    },
    errors::LangExplorerError,
    expanders::{
        EmbedderWrapper, FrontierDecisionWrapper, GrammarExpander, ProductionDecisionWrapper,
    },
    grammar::{
        grammar::Grammar,
        lhs::ProductionLHS,
        prod::Production,
        program::{ProgramInstance, WLKernelHashingOrder},
        rule::ProductionRule,
        NonTerminal, Terminal,
    },
    tooling::{
        modules::expander::{
            frontier_decision::FrontierDecisionAttentionConfig,
            prod_decision_fixed::ProductionDecisionFixedConfig, Activation,
        },
        training::TrainingParams,
    },
};

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

/// The different strategies for normalizing the output logits
/// given by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationStrategy {
    Softmax,
    LogSoftmax,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub enum LabelExtractionStrategy {
    WLKernel {
        /// Number of iterations to run to extract words.
        iterations: u32,
        /// The order in which to hash items when computing new labels.
        order: WLKernelHashingOrder,
        /// Toggle whether to deduplicate words when extracting WL kernel features.
        dedup: bool,
        /// Toggle whether to sort words when extracting WL kernel features.
        sort: bool,
    },
    CodePaths {},
}

impl Default for LabelExtractionStrategy {
    fn default() -> Self {
        Self::WLKernel {
            iterations: 5,
            order: WLKernelHashingOrder::ParentSelfChildrenOrdered,
            dedup: false,
            sort: false,
        }
    }
}

pub struct LearnedExpander<T: Terminal, I: NonTerminal, B: AutodiffBackend> {
    /// Model for embedding partial programs.
    embedder: EmbedderWrapper<T, I, B>,

    /// Wrapper for production decision model.
    production_decision: ProductionDecisionWrapper<B>,

    /// Wrapper for frontier decision model.
    frontier_decision: FrontierDecisionWrapper<B>,

    /// The strategy for extracting labels from the current program instance.
    label_extraction: LabelExtractionStrategy,

    /// Strategy for normalizing the output logits from the model.
    normalization: NormalizationStrategy,
    /// Strategy for sampling from the output distribution of the model.
    sampling: SamplingStrategy,
    /// Activation function to use on the output of the model.
    activation: Activation,
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> GrammarExpander<T, I>
    for LearnedExpander<T, I, B>
{
    fn init(grammar: &Grammar<T, I>, seed: u64) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        B::seed(seed);

        let device = Default::default();

        let d2v = Doc2VecEmbedderDBOWNS::new(
            grammar,
            Doc2VecDBOWNSEmbedderParams::new(
                AdamWConfig::new(),
                1000,
                1000,
                GeneralEmbeddingTrainingParams::new(TrainingParams::default()),
                "".to_string(),
            ),
            device,
        );

        let device = Default::default();

        let symbols = grammar.get_all_symbols();

        let production_decision = ProductionDecisionWrapper::ProdDecisionFixed(
            ProductionDecisionFixedConfig::new(256, 128).init(grammar, &device),
        );

        let frontier_decision = FrontierDecisionWrapper::FrontierDecisionV1(
            FrontierDecisionAttentionConfig::new(256, symbols.len(), 16).init(&device),
        );

        Ok(Self {
            embedder: EmbedderWrapper::Doc2Vec(d2v),
            production_decision,
            frontier_decision,
            label_extraction: LabelExtractionStrategy::WLKernel {
                iterations: 5,
                order: WLKernelHashingOrder::ParentSelfChildrenOrdered,
                dedup: true,
                sort: false,
            },
            normalization: NormalizationStrategy::Softmax,
            sampling: SamplingStrategy::HighestProb,
            activation: Activation::ReLU,
        })
    }

    fn expand_rule<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        context: &'a ProgramInstance<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        let doc = context.clone();

        let words = match &self.label_extraction {
            LabelExtractionStrategy::WLKernel {
                iterations,
                order,
                dedup,
                sort,
            } => doc.extract_words_wl_kernel(*iterations, order.clone(), *dedup, *sort),
            LabelExtractionStrategy::CodePaths {} => {
                panic!("CodePaths label extraction not implemented yet")
            }
        };

        let embedding = self
            .embedder
            .forward(doc, words)
            .expect("failed to create embedding");

        let out = self.production_decision.forward(
            vec![production],
            embedding.unsqueeze_dim(0),
            self.normalization.clone(),
            self.sampling.clone(),
            self.activation.clone(),
        );

        let idx = *out
            .to_data()
            .convert::<u64>()
            .to_vec::<u64>()
            .expect("failed to convert tensor to vec")
            .first()
            .expect("failed to get index") as usize;

        production
            .get(idx)
            .expect("couldn't find the selected index")
    }

    fn choose_lhs_and_slot<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        _context: &'a ProgramInstance<T, I>,
        lhs_location_matrix: &[(&'a ProductionLHS<T, I>, Vec<usize>)],
    ) -> (&'a ProductionLHS<T, I>, usize) {
        todo!()
    }

    fn cleanup(&mut self) {}
}
