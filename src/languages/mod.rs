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

use std::collections::HashSet;
use std::{fmt::Display, str::FromStr, time::SystemTime};

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{channel, Receiver, Sender};
use utoipa::ToSchema;

use crate::grammar::program::{InstanceId, WLKernelHashingOrder};
use crate::{
    errors::LangExplorerError,
    evaluators::Evaluator,
    expanders::ExpanderWrapper,
    grammar::{grammar::Grammar, BinarySerialize, NonTerminal, Terminal},
    languages::{
        css::{CSSLanguage, CSSLanguageParameters},
        nft_ruleset::{NFTRulesetLanguage, NFTRulesetParams},
        spice::{SpiceLanguage, SpiceLanguageParams},
        spiral::{SpiralLanguage, SpiralLanguageParams},
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

    /// Toggle whether or not to return partial graphs for each program
    /// that is generated.
    #[serde(alias = "return_partial_graphs", default = "default_return_partials")]
    return_partial_graphs: bool,

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

    /// Specify the number of WL-kernel iterations to be run
    /// with each graph to extract features.
    #[serde(alias = "wl_degree", default = "default_wl_degree")]
    wl_degree: u32,
}

fn default_count() -> u64 {
    1
}

fn default_wl_degree() -> u32 {
    3
}

fn default_return_partials() -> bool {
    false
}

impl GenerateParams {
    pub async fn execute(
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

        let mut programs = HashSet::new();
        let mut features = vec![];
        let mut edge_lists = vec![];
        let mut partial_programs = HashSet::new();

        let grammar_fmt;
        if self.return_grammar {
            grammar_fmt = Some(format!("{}", &grammar));
        } else {
            grammar_fmt = None;
        }

        if self.count > 0 {
            let start = SystemTime::now();
            let num_cpus = num_cpus::get() as u64;

            let size = match self.count / num_cpus {
                0 => 1,
                o => o,
            } as usize;

            let (tx, mut rx): (Sender<ProgramResult>, Receiver<ProgramResult>) = channel(size);

            for _ in 0..num_cpus {
                let exp = expander.clone();
                let gc = grammar.clone();
                let txt = tx.clone();
                tokio::spawn(async move {
                    let mut expander = exp.get_expander().unwrap();
                    let count = self.count / num_cpus; // TODO fix

                    for _ in 0..count {
                        let mut res = ProgramResult::new();
                        match Grammar::generate_program_instance(&gc, &mut expander) {
                            Ok(prog) => {
                                if self.return_features {
                                    res.set_features(prog.extract_words_wl_kernel(
                                        self.wl_degree,
                                        WLKernelHashingOrder::SelfChildrenParentOrdered,
                                    ));
                                }

                                if self.return_edge_lists {
                                    res.set_edge_list(prog.get_edge_list());
                                }

                                if self.return_partial_graphs {
                                    res.set_partial_programs(prog.get_all_sub_programs());
                                }

                                match String::from_utf8(prog.serialize()) {
                                    Ok(data) => res.set_program(data),
                                    Err(e) => return Err(e.into()),
                                }
                            }
                            Err(e) => return Err(e),
                        }

                        // Ship it off and keep going.
                        txt.send(res).await.unwrap();
                    }

                    Ok(())
                });
            }

            // Need to drop the original sender since we don't move it over to a task.
            drop(tx);

            while let Some(res) = rx.recv().await {
                // Only add new elements to the set of programs, automatic deduplication.
                if let Some(prog) = res.program {
                    if programs.insert(prog) {
                        if let Some(feat) = res.features {
                            features.push(feat);
                        }

                        if let Some(edges) = res.edge_list {
                            edge_lists.push(edges);
                        }

                        if let Some(mut partials) = res.partial_programs {
                            for prog in partials.drain() {
                                partial_programs.insert(prog);
                            }
                        }
                    }
                }
            }

            let elapsed = start.elapsed().unwrap();

            println!(
                "generated {} programs and {} features for language {language} with expander {expander} in {} seconds",
                programs.len(),
                features.len(),
                elapsed.as_secs()
            );
        }

        Ok(GenerateResults {
            grammar: grammar_fmt,
            programs: programs,
            partial_programs,
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
    programs: HashSet<String>,

    /// If configured,
    #[serde(alias = "partial_programs")]
    partial_programs: HashSet<String>,

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

struct ProgramResult {
    /// If enabled, the string representation of the generated program.
    program: Option<String>,

    /// If enabled, returns all deduplicated sub-programs that are
    /// created from the main generated program.
    partial_programs: Option<HashSet<String>>,

    /// If enabled, returns a list of all features extracted from
    /// the program.
    features: Option<Vec<u64>>,

    /// If enabled, returns the program graph in edge-list foramt.
    edge_list: Option<Vec<(InstanceId, InstanceId)>>,
}

impl ProgramResult {
    fn new() -> Self {
        Self {
            program: None,
            partial_programs: None,
            features: None,
            edge_list: None,
        }
    }

    fn set_program(&mut self, program: String) {
        self.program = Some(program);
    }

    fn set_partial_programs(&mut self, programs: HashSet<String>) {
        self.partial_programs = Some(programs);
    }

    fn set_features(&mut self, features: Vec<u64>) {
        self.features = Some(features);
    }

    fn set_edge_list(&mut self, edge_list: Vec<(InstanceId, InstanceId)>) {
        self.edge_list = Some(edge_list);
    }
}
