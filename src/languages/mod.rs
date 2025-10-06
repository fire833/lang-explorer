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

use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::Arc;
use std::{fmt::Display, str::FromStr, time::SystemTime};

use bhtsne::tSNE;
#[allow(unused)]
use burn::backend::{Autodiff, Cuda, NdArray};
use burn::data::dataset::Dataset;
use burn::optim::AdamWConfig;
use burn::prelude::Backend;
use clap::ValueEnum;
use dashmap::DashMap;
use reqwest::ClientBuilder;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{channel, Receiver, Sender};
use utoipa::ToSchema;

use crate::embedding::doc2vecdbowns::{Doc2VecDBOWNSEmbedderParams, Doc2VecEmbedderDBOWNS};
use crate::embedding::{GeneralEmbeddingTrainingParams, LanguageEmbedder};
use crate::expanders::learned::LabelExtractionStrategy;
use crate::grammar::prod::Production;
use crate::grammar::program::{InstanceId, ProgramInstance};
use crate::languages::anbncn::{AnBnCnLanguage, AnBnCnLanguageParams};
use crate::languages::karel::{KarelLanguage, KarelLanguageParameters};
use crate::languages::strings::StringValue;
use crate::tooling::d2v::{self};
use crate::tooling::ollama::get_embedding_ollama;
use crate::tooling::similarity::{vector_similarity, wl_test, VectorSimilarity};
use crate::{
    errors::LangExplorerError,
    evaluators::Evaluator,
    expanders::ExpanderWrapper,
    grammar::{grammar::Grammar, NonTerminal, Terminal},
    languages::{
        css::{CSSLanguage, CSSLanguageParameters},
        nft_ruleset::{NFTRulesetLanguage, NFTRulesetParams},
        spice::{SpiceLanguage, SpiceLanguageParams},
        spiral::{SpiralLanguage, SpiralLanguageParams},
        taco_expression::{TacoExpressionLanguage, TacoExpressionLanguageParams},
        taco_schedule::{TacoScheduleLanguage, TacoScheduleLanguageParams},
    },
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

#[derive(Debug, Clone, ValueEnum, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum SimilarityCheck {
    #[serde(alias = "basic_average")]
    BasicAverage,
}

#[derive(Debug, Clone, ValueEnum, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingModel {
    #[serde(alias = "doc2vecdbow")]
    Doc2VecDBOW,
    #[serde(alias = "docvecgensim")]
    Doc2VecGensim,
    #[serde(alias = "mxbai-embed-large")]
    MXBAILarge,
    #[serde(alias = "nomic-embed-text")]
    NomicEmbed,
    #[serde(alias = "snowflake-arctic-embed:137m")]
    SnowflakeArctic137,
    #[serde(alias = "snowflake-arctic-embed")]
    SnowflakeArctic,
    #[serde(alias = "snowflake-arctic-embed2")]
    SnowflakeArctic2,
}

impl Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Doc2VecDBOW => write!(f, "doc2vecdbow"),
            Self::Doc2VecGensim => write!(f, "doc2vecgensim"),
            Self::MXBAILarge => write!(f, "mxbai-embed-large"),
            Self::NomicEmbed => write!(f, "nomic-embed-text"),
            Self::SnowflakeArctic137 => write!(f, "snowflake-arctic-embed:137m"),
            Self::SnowflakeArctic => write!(f, "snowflake-arctic-embed"),
            Self::SnowflakeArctic2 => write!(f, "snowflake-arctic-embed2"),
        }
    }
}

impl FromStr for EmbeddingModel {
    type Err = LangExplorerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "doc2vec" | "d2vdbow" | "doc2vecdbow" | "doc2vecDBOW" => Ok(Self::Doc2VecDBOW),
            "doc2vecgensim" | "d2vgensim" | "doc2vec-gensim" => Ok(Self::Doc2VecGensim),
            "mxbailarge" | "mxbai-large" | "mxbai-embed-large" => Ok(Self::MXBAILarge),
            "nomic" | "nomic-embed-text" => Ok(Self::NomicEmbed),
            "snowflake-arctic-embed" => Ok(Self::SnowflakeArctic),
            "snowflake-arctic-embed2" => Ok(Self::SnowflakeArctic2),
            "snowflake-arctic-embed137" => Ok(Self::SnowflakeArctic137),
            _ => Err(LangExplorerError::General(
                "invalid embedding model value provided".into(),
            )),
        }
    }
}

impl EmbeddingModel {
    async fn create_embeddings<B: Backend>(
        &self,
        grammar: &Grammar<StringValue, StringValue>,
        params: GeneralEmbeddingTrainingParams,
        models_dir: String,
        ollama_host: String,
        d2v_host: String,
        res: &mut GenerateResultsV2,
    ) -> Result<(), LangExplorerError> {
        let emb_name = self.to_string();

        match &self {
            Self::Doc2VecDBOW => {
                // Hack for now, recompute the number of words, even
                // though this is going to be done again.
                let mut set = HashSet::new();
                let mut documents = vec![];

                for prog in res.programs.iter() {
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

                let dim = params.d_model;
                let epochs = params.get_num_epochs();
                let params = Doc2VecDBOWNSEmbedderParams::new(
                    AdamWConfig::new(),
                    set.len(),
                    res.programs.len(),
                    params,
                    models_dir,
                );

                println!(
                            "training embeddings, there are {} documents being learned and {} total words being used",
                            documents.len(),
                            set.len(),
                        );

                let start = SystemTime::now();

                let model: Doc2VecEmbedderDBOWNS<StringValue, StringValue, Autodiff<B>> =
                    Doc2VecEmbedderDBOWNS::<StringValue, StringValue, Autodiff<B>>::new(
                        grammar,
                        params,
                        Default::default(),
                    )
                    .fit(&documents)?;

                let end = start.elapsed().unwrap();

                println!(
                    "trained {} doc (and {} word) embeddings with {} epochs in {} seconds.",
                    documents.len(),
                    set.len(),
                    epochs,
                    end.as_secs()
                );

                let mut embeddings = model.get_embeddings()?;
                for prog in res.programs.iter_mut() {
                    prog.set_embedding(emb_name.clone(), embeddings.drain(0..dim).collect());
                }
            }
            Self::Doc2VecGensim => {
                let client = ClientBuilder::new()
                    .pool_max_idle_per_host(10)
                    .build()
                    .unwrap();

                let mut docs = HashMap::new();

                for prog in res.programs.iter() {
                    docs.insert(
                        prog.program.clone().unwrap(),
                        prog.features.clone().unwrap(),
                    );
                }

                match d2v::get_embedding_d2v(&client, &d2v_host, docs, &params.clone().into()).await
                {
                    Ok(resp) => {
                        for prog in res.programs.iter_mut() {
                            let vec = resp.get(&prog.program.clone().unwrap()).unwrap();
                            prog.set_embedding(emb_name.clone(), vec.clone());
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
            Self::MXBAILarge
            | Self::NomicEmbed
            | Self::SnowflakeArctic
            | Self::SnowflakeArctic2
            | Self::SnowflakeArctic137 => {
                let client = ClientBuilder::new()
                    .pool_max_idle_per_host(10)
                    .build()
                    .unwrap();

                let blank: String = "".into();

                let programs: Vec<String> = res
                    .programs
                    .iter()
                    .map(|item| match &item.program {
                        Some(prompt) => prompt.clone(),
                        None => blank.clone(),
                    })
                    .collect();

                for (idx, prog) in programs.iter().enumerate() {
                    match get_embedding_ollama(&client, &ollama_host, prog, self.clone()).await {
                        Ok(vec) => {
                            let p = res.programs.get_mut(0).unwrap();
                            p.set_embedding(emb_name.clone(), vec);
                        }
                        Err(e) => return Err(e),
                    }

                    if idx % 100 == 0 {
                        println!(
                            "processed {} / {} prompts for embeddings",
                            idx,
                            programs.len()
                        );
                    }
                }

                // match get_embeddings_bulk_ollama(&client, &ollama_host, prompts, self.clone(), 7)
                //     .await
                // {
                //     Ok(mut responses) => {
                //         for (idx, vec) in responses.iter_mut().enumerate() {
                //             let p = res.programs.get_mut(idx).unwrap();
                //             let new = mem::take(vec);
                //             p.set_embedding(emb_name.clone(), new);
                //         }
                //     }
                //     Err(e) => return Err(e),
                // }
            }
        };

        Ok(())
    }
}

/// Parameters supplied to the API for generating one or more programs with the provided
/// language and expander to create said program.
#[derive(Debug, Serialize, Deserialize, ToSchema, Clone, Default)]
pub struct GenerateParams {
    /// Toggle whether to return WL-kernel extracted features
    /// along with each graph.
    #[serde(alias = "return_features", default)]
    return_features: bool,

    /// Toggle whether to return edge lists along with each graph.
    #[serde(alias = "return_edge_lists", default)]
    return_edge_lists: bool,

    /// Toggle which embeddings to return for each instance.
    #[serde(alias = "return_embeddings", default)]
    return_embeddings: Vec<EmbeddingModel>,

    /// Which similarity experiments to run between embeddings.
    #[serde(alias = "similarity_experiments", default)]
    similarity_experiments: Vec<SimilarityCheck>,

    /// Toggle whether to return 2D t-SNE projections of each embedding.
    #[serde(alias = "return_tsne2d", default)]
    return_tsne2d: bool,

    /// Toggle whether to return 3D t-SNE projections of each embedding.
    #[serde(alias = "return_tsne3d", default)]
    return_tsne3d: bool,

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

    /// Specify the label extraction strategy for creating
    /// labels for each document program.
    #[serde(alias = "label_extraction", default)]
    label_extraction: LabelExtractionStrategy,

    /// General purpose training parameters for doing training runs.
    #[serde(flatten)]
    params: GeneralEmbeddingTrainingParams,
}

fn default_count() -> u64 {
    1
}

fn default_return_partials() -> bool {
    false
}

impl GenerateParams {
    pub async fn execute<B: Backend>(
        self,
        language: LanguageWrapper,
        expander: ExpanderWrapper,
        models_dir: String,
        ollama_host: String,
        d2v_host: String,
        _output_dir: String,
    ) -> Result<GenerateResultsV2, LangExplorerError> {
        let res_copy = self.clone();

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
            LanguageWrapper::AnBnCn => AnBnCnLanguage::generate_grammar(AnBnCnLanguageParams {}),
        }?;

        let mut results = GenerateResultsV2 {
            grammar: None,
            programs: vec![],
            options: res_copy,
            language: language.clone(),
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
                let seed = self.params.get_seed();
                let label_extraction = self.label_extraction.clone();

                tokio::spawn(async move {
                    let mut expander = exp.get_expander(&gc, seed + i * 5).unwrap();
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
                                        &label_extraction.clone(),
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
                                                &label_extraction.clone(),
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
                            Err(e) => {
                                if gc.is_context_sensitive() {
                                    println!("error generating program: {e}");
                                } else {
                                    return Err(e);
                                }
                            }
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

            for embed in self.return_embeddings.iter() {
                println!("creating embeddings with {} model", embed);
                embed
                    .create_embeddings::<B>(
                        &grammar,
                        self.params.clone(),
                        models_dir.clone(),
                        ollama_host.clone(),
                        d2v_host.clone(),
                        &mut results,
                    )
                    .await?;
            }
        }

        Ok(results)
    }

    pub async fn from_file(path: &str) -> Result<Self, LangExplorerError> {
        let contents = fs::read_to_string(path)?;
        let config: GenerateParams = serde_json::from_str(&contents)?;
        Ok(config)
    }

    pub async fn from_experiment_id<P: Display>(
        path: P,
        lang: &LanguageWrapper,
        exp_id: usize,
    ) -> Result<Self, LangExplorerError> {
        let path = format!("{path}/{lang}/{exp_id}/options.json");
        Self::from_file(&path).await
    }
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

    /// Return the params that were used to generate these programs.
    /// Mostly just for bookkeeping & keeping good records of experiments run.
    #[serde(alias = "options")]
    options: GenerateParams,

    /// The language that was used to generate these programs.
    #[serde(alias = "language")]
    language: LanguageWrapper,
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

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct ProgramRecord {
    idx: usize,
    program: String,
    is_partial: bool,
}

impl ProgramRecord {
    fn new(idx: usize, program: String, is_partial: bool) -> Self {
        Self {
            idx,
            program,
            is_partial,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct GraphvizRecord {
    idx: usize,
    graphviz: String,
}

impl GraphvizRecord {
    fn new(idx: usize, graphviz: String) -> Self {
        Self { idx, graphviz }
    }
}

impl GenerateResultsV2 {
    pub fn write<P: Display>(&self, path: P) -> Result<(), LangExplorerError> {
        let exp_id = Self::get_experiment_id(&path, &self.language)?;

        // Fix, need to create directories here too
        fs::create_dir_all(format!("{path}/{}/{exp_id}", self.language))?;

        let mut program_writer =
            csv::Writer::from_path(format!("{path}/{}/{exp_id}/programs.csv", self.language))?;

        println!(
            "writing {} programs to {path}/{}/{exp_id}/programs.csv",
            self.programs.len(),
            self.language
        );
        for (idx, prog) in self.programs.iter().enumerate() {
            program_writer.serialize(ProgramRecord::new(
                idx,
                prog.program.clone().unwrap_or("".into()),
                prog.is_partial,
            ))?;
        }

        program_writer.flush()?;

        // Need to clone to avoid immutable and mutable borrow downstream.
        for embed_model in self.options.return_embeddings.clone().iter() {
            println!(
                "writing embeddings for model {} to {path}/{}/{exp_id}/embeddings_{}.csv",
                embed_model, self.language, embed_model
            );
            let mut embed_writer = csv::Writer::from_path(format!(
                "{path}/{}/{exp_id}/embeddings_{}.csv",
                self.language, embed_model
            ))?;

            for (idx, prog) in self.programs.iter().enumerate() {
                if let Some(map) = &prog.embeddings {
                    if let Some(emb) = map.get(&embed_model.to_string()) {
                        if idx == 0 {
                            let mut vec = vec!["idx".to_string()];
                            for i in 0..emb.len() {
                                vec.push(format!("dim_{}", i));
                            }

                            embed_writer.write_record(vec)?;
                        }

                        embed_writer.write_field(idx.to_string())?;

                        for val in emb.iter() {
                            embed_writer.write_field(val.to_string())?;
                        }

                        embed_writer.write_record(None::<&[u8]>)?;
                    }
                }

                // Flush every once in a while.
                if idx % 1000 == 0 {
                    embed_writer.flush()?;
                }
            }

            if self.options.return_tsne2d {
                println!(
                    "creating 2D t-SNE projections for model {} to {path}/{}/{exp_id}/tsne2d.csv",
                    embed_model, self.language
                );
                self.create_tsne(
                    embed_model.to_string(),
                    format!("{path}/{}/{exp_id}/tsne2d.csv", self.language),
                    2,
                );
            }

            if self.options.return_tsne3d {
                println!(
                    "creating 3D t-SNE projections for model {} to {path}/{}/{exp_id}/tsne3d.csv",
                    embed_model, self.language
                );
                self.create_tsne(
                    embed_model.to_string(),
                    format!("{path}/{}/{exp_id}/tsne3d.csv", self.language),
                    3,
                );
            }

            embed_writer.flush()?;
        }

        if self.options.return_graphviz {
            let mut graphviz_writer =
                csv::Writer::from_path(format!("{path}/{}/{exp_id}/graphviz.csv", self.language))?;

            println!(
                "writing graphviz for {} programs to {path}/{}/{exp_id}/graphviz.csv",
                self.programs.len(),
                self.language
            );

            for (idx, prog) in self.programs.iter().enumerate() {
                graphviz_writer.serialize(GraphvizRecord::new(
                    idx,
                    prog.graphviz.clone().unwrap_or("".into()),
                ))?;
            }

            graphviz_writer.flush()?;
        }

        if self.options.return_features && self.options.similarity_experiments.len() > 0 {
            let mut ast_similarity_scores = HashMap::new();

            // For all experiments, we need to compute pairwise similarities between all ASTs.
            for (i, this) in self.programs.iter().enumerate() {
                for (j, other) in self.programs.iter().skip(i + 1).enumerate() {
                    match (&this.features, &other.features) {
                        (Some(vec1), Some(vec2)) => {
                            let tuple = match (i, j) {
                                (a, b) if a <= b => (a, b),
                                (a, b) if a > b => (b, a),
                                _ => continue,
                            };

                            ast_similarity_scores
                                .insert(tuple, wl_test(vec1, vec2, VectorSimilarity::Euclidean));
                        }
                        _ => {
                            panic!("we should have features here");
                        }
                    }
                }
            }

            // Now, go through all embeddings and compute the similarity matrix.
            for emb in self.options.return_embeddings.iter() {
                for (i, this) in self.programs.iter().enumerate() {
                    for (j, other) in self.programs.iter().skip(i + 1).enumerate() {
                        match (&this.embeddings, &other.embeddings) {
                            (Some(map1), Some(map2)) => {
                                let vec1 = map1.get(&emb.to_string());
                                let vec2 = map2.get(&emb.to_string());

                                if let (Some(v1), Some(v2)) = (vec1, vec2) {
                                    let tuple = match (i, j) {
                                        (a, b) if a <= b => (a, b),
                                        (a, b) if a > b => (b, a),
                                        _ => continue,
                                    };

                                    let sim =
                                        vector_similarity(v1, v2, VectorSimilarity::Euclidean);

                                    if let Some(ast_sim) = ast_similarity_scores.get(&tuple) {
                                        println!(
                                            "AST similarity between programs {} and {} is {}, embedding similarity with model {} is {}",
                                            i,
                                            j,
                                            ast_sim,
                                            emb,
                                            sim
                                        );
                                    }
                                }
                            }
                            _ => {
                                panic!("we should have embeddings here");
                            }
                        }
                    }
                }
            }

            for sim_check in self.options.similarity_experiments.iter() {
                match sim_check {
                    SimilarityCheck::BasicAverage => {
                        for prog in self.programs.iter() {
                            if let Some(map) = &prog.embeddings {
                                for (name, emb) in map.iter() {
                                    let avg: f32 = emb.iter().sum::<f32>() / emb.len() as f32;
                                    println!(
                                        "program {} with embedding model {} has average embedding value of {}",
                                        prog.program.clone().unwrap_or("".into()),
                                        name,
                                        avg
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(grammar) = &self.grammar {
            fs::write(
                format!("{path}/{}/{exp_id}/grammar.bnf", self.language),
                grammar,
            )?;
        }

        fs::write(
            format!("{path}/{}/{exp_id}/options.json", self.language),
            serde_json::to_string_pretty(&self.options)?,
        )?;

        Ok(())
    }

    fn create_tsne(&self, embedding_name: String, output_path: String, dim: u8) {
        let mut vecs = vec![];

        for prog in self.programs.iter() {
            if let Some(map) = &prog.embeddings {
                let emb = map.get(&embedding_name).unwrap();
                vecs.push(emb.as_slice());
            }
        }

        match tSNE::new(&vecs)
            .embedding_dim(dim)
            .exact(|v1, v2| {
                let lsum = v1.iter().zip(v2.iter()).fold(0.0, |sum, (x1, x2)| {
                    let diff = x1 - x2;
                    sum + (diff * diff) as f32
                });

                lsum.sqrt()
            })
            .write_csv(&output_path)
        {
            Ok(_) => println!("wrote t-SNE results to {output_path}"),
            Err(_) => todo!(),
        }
    }

    /// Get the latest experiment ID from the output directory.
    /// If there is none, start with 1.
    fn get_experiment_id<P: Display>(
        path: P,
        language: &LanguageWrapper,
    ) -> Result<usize, LangExplorerError> {
        let dir = format!("{path}/{language}/");
        let mut max_id = 0;

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        if let Some(name) = entry.file_name().to_str() {
                            if let Ok(id) = name.parse::<usize>() {
                                if id > max_id {
                                    max_id = id;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(max_id + 1)
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub(crate) struct ProgramResult {
    /// If enabled, the string representation of the generated program.
    #[serde(alias = "program")]
    program: Option<String>,

    /// Internal representation of the generated program.
    // #[serde(skip_serializing, skip_deserializing)]
    // program_internal: Option<ProgramInstance<StringValue, StringValue>>,

    /// Optional graphviz representation for the generated program.
    #[serde(alias = "graphviz")]
    graphviz: Option<String>,

    /// If enabled, returns a list of all features extracted from
    /// the program.
    #[serde(alias = "features")]
    features: Option<Vec<Feature>>,

    /// If enabled, returns the embedding of the program.
    #[serde(alias = "embeddings")]
    embeddings: Option<HashMap<String, Vec<f32>>>,

    /// If enabled, returns t-SNE embeddings for each embedding vector.
    #[serde(alias = "tsne_2d")]
    tnse_2d: Option<HashMap<String, Vec<f32>>>,

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
            // program_internal: None,
            graphviz: None,
            features: None,
            embeddings: None,
            tnse_2d: None,
            edge_list: None,
            is_partial: false,
        }
    }

    pub(crate) fn set_is_partial(&mut self, is_partial: bool) {
        self.is_partial = is_partial;
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

    pub(crate) fn set_embedding(&mut self, name: String, embedding: Vec<f32>) {
        let map = self.embeddings.get_or_insert(HashMap::new());
        map.insert(name, embedding);
    }

    // pub(crate) fn set_internal_program(
    //     &mut self,
    //     program: ProgramInstance<StringValue, StringValue>,
    // ) {
    //     self.program_internal = Some(program);
    // }
}
