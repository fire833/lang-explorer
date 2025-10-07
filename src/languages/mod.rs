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

#[allow(unused)]
use burn::backend::{Autodiff, Cuda, NdArray};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::grammar::prod::Production;
use crate::grammar::program::ProgramInstance;
use crate::{
    errors::LangExplorerError,
    evaluators::Evaluator,
    grammar::{grammar::Grammar, NonTerminal, Terminal},
};

pub mod anbncn;
pub mod css;
pub mod karel;
pub mod nft_ruleset;
pub mod parsers;
pub mod spice;
pub mod spiral;
pub mod strings;
pub mod taco_expression;
pub mod taco_schedule;
pub mod toy_language;

/// A feature is a feature of a program. This is by default a u64, since
/// that is the CRC hash output I decided on as output from the WL-kernel operation.
pub type Feature = u64;

/// A language is a wrapper trait for languages. These are objects
/// that should contain both a grammar builder for constructing
/// grammars which can be expanded into programs via a GrammarExpander,
/// as well as an Evaluator for actually evaluating outputs and hopefully
/// being used to assess the efficacy of a given expander for that program.
///
/// There may in the future be a need to bind specific expanders to
/// a language, in which case this trait will probably grow, so look
/// forward to that in the future. Yay.
pub trait Language: GrammarBuilder + Evaluator {}

/// GrammarBuilder is a wrapper around types that are able to
/// construct grammars for downstream consumption.
pub trait GrammarBuilder {
    type Term: Terminal;
    type NTerm: NonTerminal;
    type Params<'de>: Default + Serialize + Deserialize<'de> + ToSchema;
    type Checker: GrammarExpansionChecker<Self::Term, Self::NTerm>;

    /// Method to actually construct a new grammar instance.
    /// Uses the input parameters to customize the grammar.
    fn generate_grammar<'de>(
        params: Self::Params<'de>,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError>;

    /// Create a new verifier for verifying that expansions are valid.
    fn new_checker() -> Self::Checker;
}

pub trait GrammarExpansionChecker<T: Terminal, I: NonTerminal> {
    /// Check for whether or not a given rule is valid given
    /// the context. This has originally been implemented for keeping
    /// track of context in a more tractable way than with explicit
    /// context in the grammar productions itself. Default implementation
    /// is to always return true.
    fn check<'a>(
        &mut self,
        _context: &'a ProgramInstance<T, I>,
        _production: &'a Production<T, I>,
    ) -> bool {
        true
    }
}

/// Enumeration of all supported languages currently within lang-explorer.
/// This will almost certainly grow and change with time.
#[derive(Debug, Clone, ValueEnum, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum LanguageWrapper {
    #[clap(name = "css")]
    CSS,
    #[clap(name = "nft")]
    NFT,
    #[clap(name = "spiral")]
    Spiral,
    #[clap(name = "tacoexpr")]
    TacoExpression,
    #[clap(name = "tacosched")]
    TacoSchedule,
    #[clap(name = "spice")]
    Spice,
    #[clap(name = "karel")]
    Karel,
    #[clap(name = "anbncn")]
    AnBnCn,
}

impl FromStr for LanguageWrapper {
    type Err = LangExplorerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "css" => Ok(Self::CSS),
            "nft" => Ok(Self::NFT),
            "spiral" => Ok(Self::Spiral),
            "tacoexpr" => Ok(Self::TacoExpression),
            "tacosched" => Ok(Self::TacoSchedule),
            "spice" => Ok(Self::Spice),
            "karel" => Ok(Self::Karel),
            "anbncn" => Ok(Self::AnBnCn),
            _ => Err(LangExplorerError::General(
                "invalid language value provided".into(),
            )),
        }
    }
}

impl Display for LanguageWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CSS => write!(f, "css"),
            Self::NFT => write!(f, "nft"),
            Self::Spiral => write!(f, "spiral"),
            Self::TacoExpression => write!(f, "tacoexpr"),
            Self::TacoSchedule => write!(f, "tacosched"),
            Self::Spice => write!(f, "spice"),
            Self::Karel => write!(f, "karel"),
            Self::AnBnCn => write!(f, "anbncn"),
        }
    }
}
