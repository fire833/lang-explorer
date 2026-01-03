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

use std::{fmt::Display, str::FromStr};

use async_trait::async_trait;
use burn::{
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Float, Int, Tensor},
};
use clap::ValueEnum;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    embedding::{
        doc2vecdbowns::Doc2VecEmbedderDBOWNS, graphmae::GraphMAEEmbedder, EmbeddingModel,
        LanguageEmbedder,
    },
    errors::LangExplorerError,
    expanders::{
        learned::{NormalizationStrategy, SamplingStrategy},
        mc::MonteCarloExpander,
        wmc::WeightedMonteCarloExpander,
    },
    grammar::{
        grammar::Grammar, lhs::ProductionLHS, prod::Production, program::ProgramInstance,
        rule::ProductionRule, BinarySerialize,
    },
    tooling::{
        modules::expander::{
            frontier_decision::FrontierDecisionAttention,
            prod_decision_attn::ProductionDecisionAttention,
            prod_decision_fixed::ProductionDecisionFixed,
            prod_decision_var::ProductionDecisionVariable, Activation,
        },
        ollama,
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
pub trait GrammarExpander: Send {
    /// We may need to initialize the expander depending on the type of grammar
    /// we are using. For example, with my ML based example, the internal models of
    /// the expander may change completely depending on the rules of the grammar
    /// I want to expand.
    fn init(grammar: &Grammar, seed: u64) -> Result<Self, LangExplorerError>
    where
        Self: Sized;

    /// We want the expander to take a grammar and the current rule and
    /// make a decision on what the next expansion should be.
    fn expand_rule<'a>(
        &mut self,
        grammar: &'a Grammar,
        context: &'a ProgramInstance,
        production: &'a Production,
    ) -> &'a ProductionRule;

    /// For context sensitive grammars, we could be in a situation where we have
    /// multiple left-hand sides that match some point on the frontier, along with
    /// multiple positions within the frontier where we could expand such left-hand side
    /// with a production. Thus, we want the expander to have the ability to make this
    /// decision on our behalf as well.
    fn choose_lhs_and_slot<'a>(
        &mut self,
        grammar: &'a Grammar,
        context: &'a ProgramInstance,
        lhs_location_matrix: &[(&'a ProductionLHS, Vec<usize>)],
    ) -> (&'a ProductionLHS, usize);

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
    pub fn get_expander(
        &self,
        grammar: &Grammar,
        seed: u64,
    ) -> Result<Box<dyn GrammarExpander>, LangExplorerError> {
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
enum EmbedderWrapper<B: AutodiffBackend> {
    Doc2Vec(Doc2VecEmbedderDBOWNS<B>),
    GraphMAE(GraphMAEEmbedder<B>),
    Ollama(EmbeddingModel),
}

impl<B: AutodiffBackend> EmbedderWrapper<B> {
    fn forward(&mut self, doc: ProgramInstance) -> Result<Tensor<B, 1, Float>, LangExplorerError> {
        match self {
            Self::Doc2Vec(d2ve) => d2ve.embed(doc),
            Self::GraphMAE(graphmae) => graphmae.embed(doc),
            Self::Ollama(model) => {
                let data = String::from_utf8(doc.serialize())?;
                let client = Client::new();

                let vec = ollama::get_embedding_ollama_sync(
                    &client,
                    // TODO: don't hardcode
                    &"https://ollama.soonerhpclab.org".into(),
                    &data,
                    model.clone(),
                )?;

                let dev = Default::default();
                let tensor = Tensor::<B, 1, Float>::from_data(vec.as_slice(), &dev);

                Ok(tensor)
            }
        }
    }
}

enum ProductionDecisionWrapper<B: Backend> {
    ProdDecisionFixed(ProductionDecisionFixed<B>),
    ProdDecisionVariable(ProductionDecisionVariable<B>),
    ProdDecisionAttentionOnly(ProductionDecisionAttention<B>),
}

impl<B: Backend> ProductionDecisionWrapper<B> {
    fn forward(
        &self,
        productions: Vec<&Production>,
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
    fn forward(&self, frontier: &[(&ProductionLHS, Vec<usize>)]) -> Tensor<B, 2, Float> {
        let dev = Default::default(); // This will probably bite me and be wrong with multiple GPUs.
        let num_rules = frontier.len();
        let matrix = frontier
            .iter()
            .flat_map(|(_, row)| row.clone())
            .collect::<Vec<usize>>();

        let tensor =
            Tensor::<B, 2, Int>::from_data(matrix.as_slice(), &dev).reshape([num_rules as i32, -1]);

        match self {
            FrontierDecisionWrapper::FrontierDecisionV1(model) => model.forward(tensor),
        }
    }
}
