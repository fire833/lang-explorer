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

use std::{fmt::Display, str::FromStr, time::SystemTime};

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::grammar::program::{InstanceId, WLKernelHashingOrder};
use crate::{
    errors::LangExplorerError,
    evaluators::Evaluator,
    expanders::{mc::MonteCarloExpander, ExpanderWrapper, GrammarExpander},
    grammar::{BinarySerialize, Grammar, NonTerminal, Terminal},
    languages::{
        css::{CSSLanguage, CSSLanguageParameters},
        nft_ruleset::{NFTRulesetLanguage, NFTRulesetParams},
        spice::{SpiceLanguage, SpiceLanguageParams},
        spiral::{SpiralLanguage, SpiralLanguageParams},
        strings::StringValue,
        taco_expression::{TacoExpressionLanguage, TacoExpressionLanguageParams},
        taco_schedule::{TacoScheduleLanguage, TacoScheduleLanguageParams},
    },
};

pub mod css;
pub mod nft_ruleset;
pub mod parsers;
pub mod spice;
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
    type Params<'de>: Default + Serialize + Deserialize<'de> + ToSchema;

    fn generate_grammar<'de>(
        params: Self::Params<'de>,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError>;
}

/// Enumeration of all supported languages currently within lang-explorer.
/// This will almost certainly grow and change with time.
#[derive(Clone, ValueEnum, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum LanguageWrapper {
    CSS,
    NFT,
    Spiral,
    TacoExpression,
    TacoSchedule,
    Spice,
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
        }
    }
}

/// Parameters supplied to the API for generating one or more programs with the provided
/// language and expander to create said program.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GenerateParams {
    /// Toggle whether to return WL-kernel extracted features
    /// along with each graph.
    #[serde(alias = "return_features", default)]
    return_features: bool,

    /// Toggle whether to return edge lists along with each graph.
    #[serde(alias = "return_edge_lists", default)]
    return_edge_lists: bool,

    /// Toggle whether to return the grammar in BNF form that was
    /// used to generate programs.
    #[serde(alias = "return_grammar", default)]
    return_grammar: bool,

    /// Parameters for CSS Language.
    #[serde(alias = "css", default)]
    css: CSSLanguageParameters,

    /// Parameters for NFTables language.
    #[serde(alias = "nft", default)]
    nft: NFTRulesetParams,

    /// Parameters for SPICE language.
    #[serde(alias = "spice", default)]
    spice: SpiceLanguageParams,

    /// Parameters for SPIRAL Language.
    #[serde(alias = "spiral", default)]
    spiral: SpiralLanguageParams,

    /// Parameters for Taco Expression Language.
    #[serde(alias = "taco_expression", default)]
    taco_expr: TacoExpressionLanguageParams,

    /// Parameters for Taco Schedule Language.
    #[serde(alias = "taco_schedule", default)]
    taco_sched: TacoScheduleLanguageParams,

    /// Specify the number of programs to generate.
    #[serde(alias = "count", default = "default_count")]
    count: u64,

    /// Specify the number of WL-kernel iterations to be run with each graph to extract features.
    #[serde(alias = "wl_degree", default = "default_wl_degree")]
    wl_degree: u32,
}

fn default_count() -> u64 {
    1
}

fn default_wl_degree() -> u32 {
    3
}

impl GenerateParams {
    pub fn execute(
        self,
        language: LanguageWrapper,
        expander: ExpanderWrapper,
    ) -> Result<GenerateResults, LangExplorerError> {
        let grammar = match language {
            LanguageWrapper::CSS => CSSLanguage::generate_grammar(self.css),
            LanguageWrapper::NFT => NFTRulesetLanguage::generate_grammar(self.nft),
            LanguageWrapper::Spiral => SpiralLanguage::generate_grammar(self.spiral),
            LanguageWrapper::TacoExpression => {
                TacoExpressionLanguage::generate_grammar(self.taco_expr)
            }
            LanguageWrapper::TacoSchedule => {
                TacoScheduleLanguage::generate_grammar(self.taco_sched)
            }
            LanguageWrapper::Spice => SpiceLanguage::generate_grammar(self.spice),
        }?;

        let mut exp: Box<dyn GrammarExpander<StringValue, StringValue>> = match expander {
            ExpanderWrapper::MonteCarlo => Box::new(MonteCarloExpander::new()),
            ExpanderWrapper::ML => {
                return Err(LangExplorerError::General(
                    "ml method not implemented".into(),
                ))
            }
        };

        let mut programs = vec![];
        let mut features = vec![];
        let mut edge_lists = vec![];

        if self.count > 0 {
            let start = SystemTime::now();

            for _ in 0..self.count {
                match grammar.generate_program_instance(&mut exp) {
                    Ok(prog) => {
                        if self.return_features {
                            features.push(prog.extract_words_wl_kernel(
                                self.wl_degree,
                                WLKernelHashingOrder::SelfChildrenParentOrdered,
                            ));
                        }

                        if self.return_edge_lists {
                            edge_lists.push(prog.get_edge_list());
                        }

                        match String::from_utf8(prog.serialize()) {
                            Ok(data) => programs.push(data),
                            Err(e) => return Err(e.into()),
                        }
                    }
                    Err(e) => return Err(e),
                }
            }

            let elapsed = start.elapsed().unwrap();

            println!(
                "generated {} programs and {} features for language {} with expander {} in {} seconds",
                programs.len(),
                features.len(),
                language,
                expander,
                elapsed.as_secs()
            );
        }

        let grammar_fmt;
        if self.return_grammar {
            grammar_fmt = Some(format!("{}", grammar));
        } else {
            grammar_fmt = None;
        }

        Ok(GenerateResults {
            grammar: grammar_fmt,
            programs: programs,
            features: features,
            edge_lists: edge_lists,
        })
    }
}

/// The results generated by the program and resturned to the user.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GenerateResults {
    /// If configured, returns the list of program strings that
    /// were generated by the API.
    #[serde(alias = "programs")]
    programs: Vec<String>,

    /// If enabled, returns a list of all features extracted from
    /// each program. Every ith array in dimension 1 corresponds
    /// to the ith program in the programs field.
    #[serde(alias = "features")]
    features: Vec<Vec<u64>>,

    /// If enabled, returns an edge list representation of every
    /// program generated by the API.
    #[serde(alias = "edge_lists")]
    edge_lists: Vec<Vec<(InstanceId, InstanceId)>>,

    /// If enabled, returns a BNF copy of the grammar that was used
    /// to generate all programs within this batch.
    #[serde(alias = "grammar")]
    grammar: Option<String>,
}
