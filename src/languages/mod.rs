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
use std::sync::Arc;
use std::{fmt::Display, str::FromStr, time::SystemTime};

use burn::backend::{Autodiff, Cuda, NdArray};
use burn::data::dataset::Dataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamWConfig;
use clap::ValueEnum;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{channel, Receiver, Sender};
use utoipa::ToSchema;

use crate::embedding::doc2vecdbow::{Doc2VecDBOWEmbedderParams, Doc2VecEmbedderDBOW};
use crate::embedding::{GeneralEmbeddingTrainingParams, LanguageEmbedder};
use crate::grammar::program::{InstanceId, WLKernelHashingOrder};
use crate::languages::karel::{KarelLanguage, KarelLanguageParameters};
use crate::languages::strings::StringValue;
use crate::tooling::modules::embed::AggregationMethod;
use crate::tooling::training::TrainingParams;
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
    Karel,
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

    /// Toggle whether to return embeddings along with each program.
    #[serde(alias = "return_embeddings", default)]
    return_embeddings: bool,

    /// Toggle whether to return the grammar in BNF form that was
    /// used to generate programs.
    #[serde(alias = "return_grammar", default)]
    return_grammar: bool,

    /// Toggle whether to return graphviz graph representations of
    /// generated programs.
    #[serde(alias = "return_graphviz", default)]
    return_graphviz: bool,

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

    /// Parameters for Karel DSL.
    #[serde(alias = "karel", default)]
    karel: KarelLanguageParameters,

    /// Specify the number of programs to generate.
    #[serde(alias = "count", default = "default_count")]
    count: u64,

    /// Specify a seed to use when running experiments to deterministically
    /// set RNGs.
    #[serde(alias = "seed", default = "default_seed")]
    seed: u64,

    /// Specify the number of WL-kernel iterations to be run
    /// with each graph to extract features.
    #[serde(alias = "wl_degree", default = "default_wl_degree")]
    wl_degree: u32,

    /// When creating embeddings, specify the learning rate to use
    /// when training the embeddings.
    #[serde(alias = "learning_rate", default = "default_learning_rate")]
    learning_rate: f64,

    /// When creating embeddings, specify the number of epochs
    /// to train with.
    #[serde(alias = "num_epochs", default = "default_epochs")]
    num_epochs: u32,

    /// When creating embeddings, specify the size of the
    /// batches to train on.
    #[serde(alias = "batch_size", default = "default_batch_size")]
    batch_size: u32,

    /// When creating embeddings, specify the desired dimension
    /// of the constructed embeddings.
    #[serde(alias = "embedding_dim", default = "default_embedding_dim")]
    embedding_dimension: u32,

    /// When creating embeddings, specify the number of negative samples
    /// to update when training new embeddings.
    #[serde(alias = "num_negative_samples", default = "default_n_neg_samples")]
    num_negative_samples: u32,

    /// When creating embeddings, specify the number of context words to
    /// the left of the center word that will be trained.
    #[serde(alias = "window_left", default = "default_window_left")]
    window_left: u32,

    /// When creating embeddings, specify the number of context words to
    /// the right of the center word that will be trained.
    #[serde(alias = "window_right", default = "default_window_right")]
    window_right: u32,

    /// Set the clipping value for gradients.
    #[serde(alias = "gradient_clip_norm", default = "default_grad_clip")]
    gradient_clip_norm: f32,
}

fn default_seed() -> u64 {
    rand::random::<u64>()
}

fn default_count() -> u64 {
    1
}

fn default_wl_degree() -> u32 {
    3
}

fn default_n_neg_samples() -> u32 {
    32
}

fn default_window_left() -> u32 {
    5
}

fn default_grad_clip() -> f32 {
    0.7
}

fn default_window_right() -> u32 {
    5
}

fn default_embedding_dim() -> u32 {
    128
}

fn default_epochs() -> u32 {
    5
}

fn default_batch_size() -> u32 {
    100
}

fn default_learning_rate() -> f64 {
    0.001
}

fn default_return_partials() -> bool {
    false
}

impl GenerateParams {
    pub async fn execute(
        self,
        language: LanguageWrapper,
        expander: ExpanderWrapper,
    ) -> Result<GenerateResultsV2, LangExplorerError> {
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
            LanguageWrapper::Karel => KarelLanguage::generate_grammar(self.karel),
        }?;

        let mut results = GenerateResultsV2 {
            grammar: None,
            programs: vec![],
        };

        if self.return_grammar {
            results.grammar = Some(format!("{}", &grammar));
        }

        let all_programs: Arc<DashMap<String, u8>> = Arc::new(DashMap::new());

        if self.count > 0 {
            let start = SystemTime::now();
            let num_cpus = num_cpus::get() as u64;

            let size = match self.count / num_cpus {
                0 => 1,
                o => o,
            } as usize;

            let (tx, mut rx): (Sender<ProgramResult>, Receiver<ProgramResult>) = channel(size);

            for i in 0..num_cpus {
                let exp = expander.clone();
                let gc = grammar.clone();
                let txt = tx.clone();
                let all_progs = all_programs.clone(); // Need to do this twice, idk
                tokio::spawn(async move {
                    let mut expander = exp.get_expander(self.seed + i * 5).unwrap();
                    let all_programs = all_progs.clone();
                    let count = self.count / num_cpus; // TODO fix

                    let mut created_count = 0;

                    while created_count < count {
                        match Grammar::generate_program_instance(&gc, &mut expander) {
                            Ok(prog) => {
                                let s = prog.to_string();
                                if !all_programs.contains_key(&s) {
                                    all_programs.insert(s, 0);
                                    created_count += 1;

                                    match prog.to_result(
                                        self.return_features,
                                        self.return_edge_lists,
                                        self.return_graphviz,
                                        true,
                                        self.wl_degree,
                                    ) {
                                        // Ship it off and keep going.
                                        Ok(res) => match txt.send(res).await {
                                            Ok(_) => {}
                                            Err(e) => return Err(e.into()),
                                        },
                                        Err(e) => return Err(e),
                                    }
                                }

                                if self.return_partial_graphs {
                                    // Go through all child nodes and try and add them to the output.
                                    for partial in prog.get_all_nodes().iter().skip(1) {
                                        let s = partial.to_string();
                                        if !all_programs.contains_key(&s) {
                                            all_programs.insert(s, 0);
                                            match partial.to_result(
                                                self.return_features,
                                                self.return_edge_lists,
                                                self.return_graphviz,
                                                false,
                                                self.wl_degree,
                                            ) {
                                                Ok(res) => match txt.send(res).await {
                                                    Ok(_) => {}
                                                    Err(e) => return Err(e.into()),
                                                },
                                                Err(e) => return Err(e),
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => return Err(e),
                        }
                    }

                    Ok(())
                });
            }

            // Need to drop the original sender since we don't move it over to a task.
            drop(tx);

            while let Some(res) = rx.recv().await {
                results.programs.push(res);
            }

            let elapsed = start.elapsed().unwrap();

            println!(
                "generated {} programs for language {language} with expander {expander} in {} seconds",
                results.programs.len(),
                elapsed.as_secs()
            );

            if self.return_embeddings && self.return_features {
                // Hack for now, recompute the number of words, even
                // though this is going to be done again.
                let mut set = HashSet::new();
                let mut documents = vec![];

                for prog in results.programs.iter() {
                    documents.push((
                        prog.program.clone().unwrap(),
                        prog.features.clone().unwrap(),
                    ));
                    if let Some(words) = &prog.features {
                        for word in words.iter() {
                            set.insert(*word);
                        }
                    }
                }

                let params = Doc2VecDBOWEmbedderParams::new(
                    AdamWConfig::new(),
                    set.len(),
                    results.programs.len(),
                    GeneralEmbeddingTrainingParams::new(
                        AggregationMethod::Average,
                        TrainingParams::new(),
                    ),
                );

                println!(
                    "training embeddings, there are {} documents being learned and {} total words being used.",
                    documents.len(),
                    set.len()
                );

                let start = SystemTime::now();

                let model: Doc2VecEmbedderDBOW<StringValue, StringValue, Autodiff<NdArray>> =
                    Doc2VecEmbedderDBOW::<StringValue, StringValue, Autodiff<NdArray>>::new(
                        &grammar,
                        params,
                        Default::default(),
                    )
                    .fit(&documents)?;

                let end = start.elapsed().unwrap();

                println!(
                    "trained {} doc (and {} word) embeddings with {} epochs in {} seconds.",
                    documents.len(),
                    set.len(),
                    self.num_epochs,
                    end.as_secs()
                );

                let mut embeddings = model.get_embeddings()?;

                for prog in results.programs.iter_mut() {
                    prog.set_embedding(
                        embeddings
                            .drain(0..self.embedding_dimension as usize)
                            .collect(),
                    );
                }
            }
        }

        Ok(results)
    }

    #[deprecated()]
    pub async fn execute_legacy(
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
            LanguageWrapper::Karel => KarelLanguage::generate_grammar(self.karel),
        }?;

        let mut programs = vec![];
        let mut features = vec![];
        let mut edge_lists = vec![];

        let grammar_fmt = if self.return_grammar {
            Some(format!("{}", &grammar))
        } else {
            None
        };

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
                    let mut expander = exp.get_expander(self.seed).unwrap();
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
                if let Some(prog) = res.program {
                    programs.push(prog);
                }

                if let Some(feat) = res.features {
                    features.push(feat);
                }

                if let Some(edges) = res.edge_list {
                    edge_lists.push(edges);
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
    features: Vec<Vec<Feature>>,

    /// If enabled, returns an edge list representation of every
    /// program generated by the API.
    #[serde(alias = "edge_lists")]
    edge_lists: Vec<Vec<(InstanceId, InstanceId)>>,

    /// If enabled, returns a BNF copy of the grammar that was used
    /// to generate all programs within this batch.
    #[serde(alias = "grammar")]
    grammar: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GenerateResultsV2 {
    /// The list of programs that have been generated.
    #[serde(alias = "programs")]
    programs: Vec<ProgramResult>,

    /// If enabled, returns a BNF copy of the grammar that was used
    /// to generate all programs within this batch.
    #[serde(alias = "grammar")]
    grammar: Option<String>,
}

impl Dataset<ProgramResult> for GenerateResultsV2 {
    fn get(&self, index: usize) -> Option<ProgramResult> {
        // This crap is going to be expensive and I shouldn't be doing it,
        // but that's a problem for later.
        self.programs.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.programs.len()
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub(crate) struct ProgramResult {
    /// If enabled, the string representation of the generated program.
    #[serde(alias = "program")]
    program: Option<String>,

    /// Optional graphviz representation for the generated program.
    #[serde(alias = "graphviz")]
    graphviz: Option<String>,

    /// If enabled, returns a list of all features extracted from
    /// the program.
    #[serde(alias = "features")]
    features: Option<Vec<Feature>>,

    /// If enabled, returns the embedding of the program.
    #[serde(alias = "embedding")]
    embedding: Option<Vec<f64>>,

    /// If enabled, returns the program graph in edge-list format.
    #[serde(alias = "edge_list")]
    edge_list: Option<Vec<(InstanceId, InstanceId)>>,

    /// Toggle whether or not the generated program is a partial program or not.
    #[serde(alias = "is_partial")]
    is_partial: bool,
}

impl ProgramResult {
    pub(crate) fn new() -> Self {
        Self {
            program: None,
            graphviz: None,
            features: None,
            embedding: None,
            edge_list: None,
            is_partial: false,
        }
    }

    pub(crate) fn set_program(&mut self, program: String) {
        self.program = Some(program);
    }

    pub(crate) fn set_graphviz(&mut self, graphviz: String) {
        self.graphviz = Some(graphviz);
    }

    pub(crate) fn set_features(&mut self, features: Vec<Feature>) {
        self.features = Some(features);
    }

    pub(crate) fn set_edge_list(&mut self, edge_list: Vec<(InstanceId, InstanceId)>) {
        self.edge_list = Some(edge_list);
    }

    pub(crate) fn set_embedding(&mut self, embedding: Vec<f64>) {
        self.embedding = Some(embedding);
    }
}
