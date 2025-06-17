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
use serde::{de::Visitor, Deserialize, Serialize};

use crate::{
    errors::LangExplorerError,
    evaluators::Evaluator,
    expanders::{mc::MonteCarloExpander, ExpanderWrapper, GrammarExpander},
    grammar::{BinarySerialize, Grammar, NonTerminal, Terminal, WLKernelHashingOrder},
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
    type Params<'de>: Default + Serialize + Deserialize<'de>;

    fn generate_grammar<'de>(
        params: Self::Params<'de>,
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

#[derive(Debug, clap::Subcommand, PartialEq)]
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

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        match GenerateSubcommand::from_str(v) {
            Ok(s) => Ok(s),
            Err(e) => Err(E::custom(e.to_string())),
        }
    }
}

impl FromStr for GenerateSubcommand {
    type Err = LangExplorerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "program" => Ok(Self::Program),
            "grammar" => Ok(Self::Grammar),
            "progwfeat" | "programwithfeat" | "pgfeat" => Ok(Self::ProgramWithFeatures),
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
    #[serde(alias = "operation", default = "default_op")]
    op: GenerateSubcommand,

    #[serde(alias = "css", default)]
    css: CSSLanguageParameters,

    #[serde(alias = "nft", default)]
    nft: NFTRulesetParams,

    #[serde(alias = "spice", default)]
    spice: SpiceLanguageParams,

    #[serde(alias = "spiral", default)]
    spiral: SpiralLanguageParams,

    #[serde(alias = "taco_expression", default)]
    taco_expr: TacoExpressionLanguageParams,

    #[serde(alias = "taco_schedule", default)]
    taco_sched: TacoScheduleLanguageParams,

    #[serde(alias = "count", default = "default_count")]
    count: u64,

    #[serde(alias = "wl_degree", default = "default_wl_degree")]
    wl_degree: u32,
}

fn default_count() -> u64 {
    1
}

fn default_wl_degree() -> u32 {
    3
}

fn default_op() -> GenerateSubcommand {
    GenerateSubcommand::Program
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

        if self.op == GenerateSubcommand::Program
            || self.op == GenerateSubcommand::ProgramWithFeatures
        {
            let start = SystemTime::now();

            for _ in 0..self.count {
                match grammar.generate_program_instance(&mut exp) {
                    Ok(prog) => {
                        if self.op == GenerateSubcommand::ProgramWithFeatures {
                            features.push(prog.extract_words_wl_kernel(
                                self.wl_degree,
                                WLKernelHashingOrder::SelfChildrenOrdered,
                            ));
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

        match self.op {
            GenerateSubcommand::Program => Ok(GenerateResults {
                grammar: None,
                programs: Some(programs),
                features: None,
            }),
            GenerateSubcommand::Grammar => Ok(GenerateResults {
                grammar: Some(format!("{}", grammar)),
                programs: None,
                features: None,
            }),
            GenerateSubcommand::ProgramWithFeatures => Ok(GenerateResults {
                grammar: None,
                programs: Some(programs),
                features: Some(features),
            }),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateResults {
    #[serde(alias = "programs")]
    programs: Option<Vec<String>>,

    #[serde(alias = "features")]
    features: Option<Vec<Vec<u64>>>,

    #[serde(alias = "grammar")]
    grammar: Option<String>,
}
