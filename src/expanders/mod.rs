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

use std::{fmt::Display, str::FromStr};

use async_trait::async_trait;
use burn::{
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Float, Int, Tensor},
};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    embedding::{doc2vecdbowns::Doc2VecEmbedderDBOWNS, LanguageEmbedder},
    errors::LangExplorerError,
    expanders::{
        learned::{NormalizationStrategy, SamplingStrategy},
        mc::MonteCarloExpander,
        wmc::WeightedMonteCarloExpander,
    },
    grammar::{
        grammar::Grammar, lhs::ProductionLHS, prod::Production, program::ProgramInstance,
        rule::ProductionRule, NonTerminal, Terminal,
    },
    languages::Feature,
    tooling::modules::expander::{
        frontier_decision::FrontierDecisionAttention,
        prod_decision_attn::ProductionDecisionAttention,
        prod_decision_fixed::ProductionDecisionFixed,
        prod_decision_var::ProductionDecisionVariable, Activation,
    },
};

pub mod exhaustive;
pub mod learned;
pub mod mc;
pub mod wmc;

/// A grammar expander is an object that is able to take a
/// current production rule, the whole of the grammar that is
/// being utilized, and is able to spit out a production rule
/// that should be utilized from the list of possible production
/// rules that are implemented by this production.
#[async_trait]
pub trait GrammarExpander<T: Terminal, I: NonTerminal>: Send {
    /// We may need to initialize the expander depending on the type of grammar
    /// we are using. For example, with my ML based example, the internal models of
    /// the expander may change completely depending on the rules of the grammar
    /// I want to expand.
    fn init(grammar: &Grammar<T, I>, seed: u64) -> Result<Self, LangExplorerError>
    where
        Self: Sized;

    /// We want the expander to take a grammar and the current rule and
    /// make a decision on what the next expansion should be.
    fn expand_rule<'a>(
        &mut self,
        grammar: &'a Grammar<T, I>,
        context: &'a ProgramInstance<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I>;

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
    ) -> (&'a ProductionLHS<T, I>, usize);

    /// Whenever a program has finished being generated, this method will be called
    /// to reset/update internal state in the expander. This is mostly going to be used
    /// in the learned expander to run backprop and update the internal models for
    /// generating the program in the first place.
    fn cleanup(&mut self);
}

/// Enumeration of all supported expanders currently within lang-explorer.
/// This will almost certainly grow and change with time.
#[derive(Clone, ValueEnum, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ExpanderWrapper {
    #[clap(name = "mc")]
    MonteCarlo,
    #[clap(name = "wmc")]
    WeightedMonteCarlo,
    #[clap(name = "learned")]
    Learned,
}

impl ExpanderWrapper {
    pub fn get_expander<T: Terminal, I: NonTerminal>(
        &self,
        grammar: &Grammar<T, I>,
        seed: u64,
    ) -> Result<Box<dyn GrammarExpander<T, I>>, LangExplorerError> {
        match self {
            ExpanderWrapper::MonteCarlo => Ok(Box::new(MonteCarloExpander::init(grammar, seed)?)),
            ExpanderWrapper::WeightedMonteCarlo => {
                Ok(Box::new(WeightedMonteCarloExpander::init(grammar, seed)?))
            }
            ExpanderWrapper::Learned => Err(LangExplorerError::General("not implemented".into())),
            // ExpanderWrapper::Learned => Ok(Box::<LearnedExpander<T, I, B>>::new(
            //     LearnedExpander::init(grammar, seed)?,
            // )),
        }
    }
}

impl FromStr for ExpanderWrapper {
    type Err = LangExplorerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mc" | "montecarlo" => Ok(Self::MonteCarlo),
            "ml" | "learned" => Ok(Self::Learned),
            "wmc" | "weightedmontecarlo" => Ok(Self::WeightedMonteCarlo),
            _ => Err(LangExplorerError::General("invalid expander string".into())),
        }
    }
}

impl Display for ExpanderWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MonteCarlo => write!(f, "montecarlo"),
            Self::Learned => write!(f, "learned"),
            Self::WeightedMonteCarlo => write!(f, "weightedmontecarlo"),
        }
    }
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

enum ProductionDecisionWrapper<B: Backend> {
    ProdDecisionFixed(ProductionDecisionFixed<B>),
    ProdDecisionVariable(ProductionDecisionVariable<B>),
    ProdDecisionAttentionOnly(ProductionDecisionAttention<B>),
}

impl<B: Backend> ProductionDecisionWrapper<B> {
    fn forward<'a, T: Terminal, I: NonTerminal>(
        &self,
        productions: Vec<&'a Production<T, I>>,
        inputs: Tensor<B, 2, Float>,
        normalization: NormalizationStrategy,
        sampling: SamplingStrategy,
        activation: Activation,
    ) -> Tensor<B, 2, Int> {
        match self {
            ProductionDecisionWrapper::ProdDecisionFixed(model) => {
                model.forward(productions, inputs, activation, normalization, sampling)
            }
            ProductionDecisionWrapper::ProdDecisionVariable(model) => {
                model.forward(productions, inputs)
            }
            ProductionDecisionWrapper::ProdDecisionAttentionOnly(model) => model.forward(),
        }
    }
}

enum FrontierDecisionWrapper<B: Backend> {
    FrontierDecisionV1(FrontierDecisionAttention<B>),
}

impl<B: Backend> FrontierDecisionWrapper<B> {
    fn forward(&self, frontier: Tensor<B, 2, Int>) -> Tensor<B, 2, Float> {
        match self {
            FrontierDecisionWrapper::FrontierDecisionV1(model) => model.forward(frontier),
        }
    }
}
