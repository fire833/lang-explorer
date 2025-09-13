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

use burn::{optim::AdamWConfig, prelude::Backend, tensor::backend::AutodiffBackend};

use crate::{
    embedding::{
        doc2vecdbowns::{Doc2VecDBOWNSEmbedderParams, Doc2VecEmbedderDBOWNS},
        GeneralEmbeddingTrainingParams, LanguageEmbedder,
    },
    errors::LangExplorerError,
    expanders::{EmbedderWrapper, GrammarExpander},
    grammar::{
        grammar::Grammar, lhs::ProductionLHS, prod::Production, program::ProgramInstance,
        rule::ProductionRule, NonTerminal, Terminal,
    },
    tooling::{
        modules::{
            embed::AggregationMethod,
            expander::{
                frontier_decision::{FrontierDecision, FrontierDecisionConfig},
                prod_decision::{ProductionDecision, ProductionDecisionConfig},
                prod_decision2::ProductionDecisionAttention,
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

enum ProductionDecisionWrapper<B: Backend> {
    ProdDecisionV1(ProductionDecision<B>),
    ProdDecisionAttentionOnly(ProductionDecisionAttention<B>),
}

enum FrontierDecisionWrapper<B: Backend> {
    FrontierDecisionV1(FrontierDecision<B>),
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> GrammarExpander<T, I>
    for FixedLearnedExpander<T, I, B>
{
    /// We may need to initialize the expander depending on the type of grammar
    /// we are using. For example, with my ML based example, the internal models of
    /// the expander may change completely depending on the rules of the grammar
    /// I want to expand.
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

        let prod_decision = ProductionDecisionWrapper::ProdDecisionV1(
            ProductionDecisionConfig::new(256, rules.len(), 16).init(&device),
        );

        let frontier_decision = FrontierDecisionWrapper::FrontierDecisionV1(
            FrontierDecisionConfig::new(256, symbols.len(), 16).init(&device),
        );

        Ok(Self {
            embedder: EmbedderWrapper::Doc2Vec(d2v),
            prod_decision,
            frontier_decision,
        })
    }

    /// We want the expander to take a grammar and the current rule and
    /// make a decision on what the next expansion should be.
    fn expand_rule<'a>(
        &mut self,
        grammar: &'a Grammar<T, I>,
        context: &'a ProgramInstance<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        todo!()
    }

    /// For context sensitive grammars, we could be in a situation where we have
    /// multiple left-hand sides that match some point on the frontier, along with
    /// multiple positions within the frontier where we could expand such left-hand side
    /// with a production. Thus, we want the expander to have the ability to make this
    /// decision on our behalf as well.
    fn choose_lhs_and_slot<'a>(
        &mut self,
        grammar: &'a Grammar<T, I>,
        context: &'a ProgramInstance<T, I>,
        lhs_location_matrix: &[(&'a ProductionLHS<T, I>, Vec<usize>)],
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
