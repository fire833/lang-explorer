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

use clap::ValueEnum;
use serde::{de::Visitor, Deserialize, Serialize};

use crate::{
    errors::LangExplorerError,
    evaluators::Evaluator,
    grammar::{Grammar, NonTerminal, Terminal},
};

pub mod css;
pub mod nft_ruleset;
pub mod parsers;
pub mod spiral;
pub mod strings;
pub mod taco_expression;
pub mod taco_schedule;
pub mod toy_language;

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
    type Params: Default;

    fn generate_grammar(
        &self,
        params: Self::Params,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError>;
}

/// Enumeration of all supported languages currently within lang-explorer.
/// This will almost certainly grow and change with time.
#[derive(Clone, ValueEnum)]
pub enum LanguageWrapper {
    CSS,
    NFT,
    Spiral,
    TacoExpression,
    TacoSchedule,
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
        }
    }
}

#[derive(Debug, clap::Subcommand)]
pub enum GenerateSubcommand {
    /// Generate one or more program instances using the given expander.
    #[command()]
    Program,

    /// Generate a BNF grammar of the given language with the given input parameters.
    #[command()]
    Grammar,

    /// Generate a new program, but also return extracted subgraphs/features
    /// for use in downstream embeddings work.
    #[command()]
    ProgramWithFeatures,
}

impl Serialize for GenerateSubcommand {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(format!("{}", self).as_str())
    }
}

impl<'de> Deserialize<'de> for GenerateSubcommand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(SubcommandVisitor)
    }
}

struct SubcommandVisitor;

impl<'de> Visitor<'de> for SubcommandVisitor {
    type Value = GenerateSubcommand;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "either of the strings 'program', 'grammar', or 'progwfeat'"
        )
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        match GenerateSubcommand::from_str(v.as_str()) {
            Ok(s) => Ok(s),
            Err(_) => Err(E::custom(
                "value must be one of 'program', 'grammar', or 'progwfeat'",
            )),
        }
    }
}

impl FromStr for GenerateSubcommand {
    type Err = LangExplorerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "program" => Ok(Self::Program),
            "grammar" => Ok(Self::Grammar),
            "progwfeat" | "programwithfeat" => Ok(Self::ProgramWithFeatures),
            _ => Err(LangExplorerError::General(
                "invalid generate operation string".into(),
            )),
        }
    }
}

impl Display for GenerateSubcommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Program => write!(f, "program"),
            Self::Grammar => write!(f, "grammar"),
            Self::ProgramWithFeatures => write!(f, "programfeatures"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateParams {
    #[serde(alias = "operation")]
    op: GenerateSubcommand,

    #[serde(alias = "count")]
    count: u64,
}
