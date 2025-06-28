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
    grammar::{grammar::Grammar, prod::Production, rule::ProductionRule, NonTerminal, Terminal},
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
pub trait GrammarExpander<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    /// We may need to initialize the expander depending on the type of grammar
    /// we are using. For example, with my ML based example, the internal models of
    /// the expander may change completely depending on the rules of the grammar
    /// I want to expand.
    fn init<'a>(grammar: &'a Grammar<T, I>) -> Result<Self, LangExplorerError>
    where
        Self: Sized;

    /// We want the expander to take a grammar and the current rule and
    /// make a decision on what the next expansion should be.
    fn expand_rule<'a>(
        &mut self,
        grammar: &'a Grammar<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I>;
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

impl FromStr for ExpanderWrapper {
    type Err = LangExplorerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mc" | "montecarlo" => Ok(Self::MonteCarlo),
            "ml" => Ok(Self::ML),
            "wmc" | "wieghtedmontecarlo" => Ok(Self::WeightedMonteCarlo),
            _ => Err(LangExplorerError::General("invalid expander string".into())),
        }
    }
}

impl Display for ExpanderWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MonteCarlo => write!(f, "montecarlo"),
            Self::ML => write!(f, "ml"),
            Self::WeightedMonteCarlo => write!(f, " weightedmontecarlo"),
        }
    }
}
