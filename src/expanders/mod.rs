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
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    errors::LangExplorerError,
    expanders::{mc::MonteCarloExpander, wmc::WeightedMonteCarloExpander},
    grammar::{
        grammar::Grammar, lhs::ProductionLHS, prod::Production, rule::ProductionRule, NonTerminal,
        Terminal,
    },
};

pub mod exhaustive;
pub mod mc;
pub mod ml;
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
        lhs_location_matrix: &Vec<(&'a ProductionLHS<T, I>, Vec<usize>)>,
    ) -> (&'a ProductionLHS<T, I>, usize);
}

/// Enumeration of all supported expanders currently within lang-explorer.
/// This will almost certainly grow and change with time.
#[derive(Clone, ValueEnum, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ExpanderWrapper {
    MonteCarlo,
    WeightedMonteCarlo,
    ML,
}

impl ExpanderWrapper {
    pub fn get_expander<T: Terminal, I: NonTerminal>(
        &self,
        seed: u64,
    ) -> Result<Box<dyn GrammarExpander<T, I>>, LangExplorerError> {
        match self {
            ExpanderWrapper::MonteCarlo => Ok(Box::new(MonteCarloExpander::new(seed))),
            ExpanderWrapper::WeightedMonteCarlo => {
                Ok(Box::new(WeightedMonteCarloExpander::new(seed)))
            }
            ExpanderWrapper::ML => Err(LangExplorerError::General(
                "ml method not implemented".into(),
            )),
        }
    }
}

impl FromStr for ExpanderWrapper {
    type Err = LangExplorerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mc" | "montecarlo" => Ok(Self::MonteCarlo),
            "ml" => Ok(Self::ML),
            "wmc" | "weightedmontecarlo" => Ok(Self::WeightedMonteCarlo),
            _ => Err(LangExplorerError::General("invalid expander string".into())),
        }
    }
}

impl Display for ExpanderWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MonteCarlo => write!(f, "montecarlo"),
            Self::ML => write!(f, "ml"),
            Self::WeightedMonteCarlo => write!(f, "weightedmontecarlo"),
        }
    }
}
