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
        grammar::Grammar, lhs::ProductionLHS, prod::Production, program::ProgramInstance,
        rule::ProductionRule, NonTerminal, Terminal,
    },
    tooling::{
        modules::{
            embed::AggregationMethod,
            expander::{
                frontier_decision::FrontierDecisionAttentionConfig,
                prod_decision_fixed::ProductionDecisionFixedConfig,
            },
        },
        training::TrainingParams,
    },
};

pub struct FixedLearnedExpander<T: Terminal, I: NonTerminal, B: AutodiffBackend> {
    /// The embedder to create embeddings for a
    /// new program or partial program.
    embedder: EmbedderWrapper<T, I, B>,

    /// Model for expansion decisions.
    prod_decision: ProductionDecisionWrapper<B>,

    /// Model for frontier expansion decisions.
    frontier_decision: FrontierDecisionWrapper<B>,
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> GrammarExpander<T, I>
    for FixedLearnedExpander<T, I, B>
{
    fn init(grammar: &Grammar<T, I>, seed: u64) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        let device = Default::default();

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

        // Hack
        let device = Default::default();

        let rules = grammar.get_all_rules();
        let symbols = grammar.get_all_symbols();

        let prod_decision = ProductionDecisionWrapper::ProdDecisionFixed(
            ProductionDecisionFixedConfig::new(256, rules.len(), 16).init(&device),
        );

        let frontier_decision = FrontierDecisionWrapper::FrontierDecisionV1(
            FrontierDecisionAttentionConfig::new(256, symbols.len(), 16).init(&device),
        );

        Ok(Self {
            embedder: EmbedderWrapper::Doc2Vec(d2v),
            prod_decision,
            frontier_decision,
        })
    }

    fn expand_rule<'a>(
        &mut self,
        grammar: &'a Grammar<T, I>,
        context: &'a ProgramInstance<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        todo!()
    }

    fn choose_lhs_and_slot<'a>(
        &mut self,
        grammar: &'a Grammar<T, I>,
        context: &'a ProgramInstance<T, I>,
        lhs_location_matrix: &[(&'a ProductionLHS<T, I>, Vec<usize>)],
    ) -> (&'a ProductionLHS<T, I>, usize) {
        todo!()
    }

    fn cleanup(&mut self) {}
}
