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

use std::{
    collections::{HashMap, HashSet, VecDeque},
    f32::{consts::E, INFINITY},
    fmt::Display,
    fs, mem,
    sync::Arc,
    time::{Duration, SystemTime},
};

use bhtsne::tSNE;
use burn::{backend::Autodiff, data::dataset::Dataset, optim::AdamWConfig, prelude::Backend};
use dashmap::DashMap;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use reqwest::ClientBuilder;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{channel, Receiver, Sender};
use utoipa::ToSchema;

use crate::{
    embedding::{
        doc2vecdbowns::{Doc2VecDBOWNSEmbedderParams, Doc2VecEmbedderDBOWNS},
        EmbeddingModel, GeneralEmbeddingTrainingParams, LanguageEmbedder,
    },
    errors::LangExplorerError,
    expanders::{learned::LabelExtractionStrategy, ExpanderWrapper},
    grammar::{
        grammar::Grammar,
        program::{InstanceId, ProgramInstance},
    },
    languages::{
        anbncn::{AnBnCnLanguage, AnBnCnLanguageParams},
        css::{CSSLanguage, CSSLanguageParameters},
        karel::{KarelLanguage, KarelLanguageParameters},
        nft_ruleset::{NFTRulesetLanguage, NFTRulesetParams},
        spice::{SpiceLanguage, SpiceLanguageParams},
        spiral::{SpiralLanguage, SpiralLanguageParams},
        taco_expression::{TacoExpressionLanguage, TacoExpressionLanguageParams},
        taco_schedule::{TacoScheduleLanguage, TacoScheduleLanguageParams},
        Feature, GrammarBuilder, LanguageWrapper,
    },
    tooling::{
        d2v,
        dist::Distribution,
        ollama::get_embeddings_bulk_ollama,
        similarity::{vector_similarity, wl_test, VectorSimilarity},
    },
};

/// Parameters supplied to the API for generating one or more programs with the provided
/// language and expander to create said program.
#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct GenerateInput {
    /// Toggle whether to return WL-kernel extracted features
    /// along with each graph.
    #[serde(rename = "return_features", default)]
    return_features: bool,

    /// Toggle whether to return edge lists along with each graph.
    #[serde(rename = "return_edge_lists", default)]
    return_edge_lists: bool,

    /// Toggle which embeddings to return for each instance.
    #[serde(rename = "return_embeddings", default)]
    return_embeddings: Vec<EmbeddingModel>,

    //// Toggle whether or not to run similarity experiments.
    #[serde(rename = "do_similarity_experiments", default)]
    do_experiments: bool,

    /// Toggle whether to return 2D t-SNE projections of each embedding.
    #[serde(rename = "return_tsne2d", default)]
    return_tsne2d: bool,

    /// Toggle whether to return 3D t-SNE projections of each embedding.
    #[serde(rename = "return_tsne3d", default)]
    return_tsne3d: bool,

    /// Toggle whether to return the grammar in BNF form that was
    /// used to generate programs.
    #[serde(rename = "return_grammar", default)]
    return_grammar: bool,

    /// Toggle whether to return graphviz graph representations of
    /// generated programs.
    #[serde(rename = "return_graphviz", default)]
    return_graphviz: bool,

    /// Toggle whether or not to return partial graphs for each program
    /// that is generated.
    #[serde(rename = "return_partial_graphs", default = "default_return_partials")]
    return_partial_graphs: bool,

    /// Parameters for CSS Language.
    #[serde(rename = "css", default)]
    css: CSSLanguageParameters,

    /// Parameters for NFTables language.
    #[serde(rename = "nft", default)]
    nft: NFTRulesetParams,

    /// Parameters for SPICE language.
    #[serde(rename = "spice", default)]
    spice: SpiceLanguageParams,

    /// Parameters for SPIRAL Language.
    #[serde(rename = "spiral", default)]
    spiral: SpiralLanguageParams,

    /// Parameters for Taco Expression Language.
    #[serde(rename = "taco_expression", default)]
    taco_expr: TacoExpressionLanguageParams,

    /// Parameters for Taco Schedule Language.
    #[serde(rename = "taco_schedule", default)]
    taco_sched: TacoScheduleLanguageParams,

    /// Parameters for Karel DSL.
    #[serde(rename = "karel", default)]
    karel: KarelLanguageParameters,

    /// Specify the number of programs to generate.
    #[serde(rename = "count", default = "default_count")]
    count: u64,

    /// Specify the number of concurrent ollama transactions to make.
    #[serde(rename = "ollama_concurrent_requests", default = "default_ollama")]
    concurrent_ollama_requests: u64,

    /// Specify the tcp keepalive time (in seconds) for ollama requests.
    #[serde(rename = "ollama_tcp_keepalive", default = "default_keepalive")]
    ollama_tcp_keepalive: u32,

    /// Specify the label extraction strategy for creating
    /// labels for each document program.
    #[serde(rename = "label_extraction", default)]
    label_extraction: LabelExtractionStrategy,

    /// General purpose training parameters for doing training runs.
    #[serde(flatten)]
    params: GeneralEmbeddingTrainingParams,
}

fn default_ollama() -> u64 {
    40
}

fn default_keepalive() -> u32 {
    3600
}

fn default_count() -> u64 {
    1
}

fn default_return_partials() -> bool {
    false
}

impl Default for GenerateInput {
    fn default() -> Self {
        Self {
            return_features: false,
            return_edge_lists: false,
            return_embeddings: vec![],
            do_experiments: false,
            return_tsne2d: false,
            return_tsne3d: false,
            return_grammar: false,
            return_graphviz: false,
            return_partial_graphs: default_return_partials(),
            css: CSSLanguageParameters::default(),
            nft: NFTRulesetParams::default(),
            spice: SpiceLanguageParams::default(),
            spiral: SpiralLanguageParams::default(),
            taco_expr: TacoExpressionLanguageParams::default(),
            taco_sched: TacoScheduleLanguageParams::default(),
            karel: KarelLanguageParameters::default(),
            count: default_count(),
            concurrent_ollama_requests: default_ollama(),
            ollama_tcp_keepalive: default_keepalive(),
            label_extraction: LabelExtractionStrategy::default(),
            params: GeneralEmbeddingTrainingParams::default(),
        }
    }
}

impl GenerateInput {
    pub async fn execute<B: Backend>(
        self,
        language: LanguageWrapper,
        expander: ExpanderWrapper,
        models_dir: String,
        ollama_host: String,
        d2v_host: String,
        _output_dir: String,
    ) -> Result<GenerateOutput, LangExplorerError> {
        let res_copy = self.clone();
        let exp_copy = self.clone();

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

        let mut results = GenerateOutput {
            grammar: None,
            programs: vec![],
            options: res_copy,
            language: language.clone(),
            similarity_experiments: None,
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
                results
                    .create_embeddings::<B>(
                        embed.clone(),
                        &grammar,
                        self.params.clone(),
                        models_dir.clone(),
                        ollama_host.clone(),
                        d2v_host.clone(),
                        self.concurrent_ollama_requests,
                        self.ollama_tcp_keepalive,
                    )
                    .await?;
            }

            if self.return_features && self.do_experiments {
                results.do_experiments(&exp_copy)?;
            }
        }

        Ok(results)
    }

    pub fn from_file(path: &str) -> Result<Self, LangExplorerError> {
        let contents = fs::read_to_string(path)?;
        let config: GenerateInput = serde_json::from_str(&contents)?;
        Ok(config)
    }

    pub fn from_experiment_id<P: Display>(
        path: P,
        lang: &LanguageWrapper,
        exp_id: usize,
    ) -> Result<Self, LangExplorerError> {
        let path = format!("{path}/{lang}/{exp_id}/options.json");
        Self::from_file(&path)
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GenerateOutput {
    /// The list of programs that have been generated.
    #[serde(rename = "programs")]
    programs: Vec<ProgramResult>,

    /// If enabled, returns a BNF copy of the grammar that was used
    /// to generate all programs within this batch.
    #[serde(rename = "grammar")]
    grammar: Option<String>,

    /// Return the params that were used to generate these programs.
    /// Mostly just for bookkeeping & keeping good records of experiments run.
    #[serde(rename = "options")]
    options: GenerateInput,

    /// Return similarity experiment results if any were run.
    #[serde(rename = "similarity_experiments")]
    similarity_experiments: Option<ExperimentResult>,

    /// The language that was used to generate these programs.
    #[serde(rename = "language")]
    language: LanguageWrapper,
}

impl Dataset<ProgramResult> for GenerateOutput {
    fn get(&self, index: usize) -> Option<ProgramResult> {
        // This crap is going to be expensive and I shouldn't be doing it,
        // but that's a problem for later.
        self.programs.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.programs.len()
    }
}

impl GenerateOutput {
    pub async fn from_experiment_id<P: Display>(
        path: P,
        lang: &LanguageWrapper,
        exp_id: usize,
    ) -> Result<Self, LangExplorerError> {
        let base_path = format!("{path}/{lang}/{exp_id}/");
        let programs_path = format!("{base_path}/programs.csv");
        let options_path = format!("{base_path}/options.json");

        let mut res = GenerateOutput {
            grammar: None,
            programs: vec![],
            options: GenerateInput::from_file(options_path.as_str())?,
            similarity_experiments: None,
            language: lang.clone(),
        };

        let mut reader = csv::Reader::from_path(programs_path)?;
        for result in reader.deserialize() {
            let record: ProgramRecord = result?;
            res.programs.push(record.into());
        }

        let dir = fs::read_dir(base_path)?;
        for entry in dir {
            match entry {
                Ok(entry) => {
                    let fname = entry.file_name().into_string().unwrap();
                    if fname.starts_with("embeddings_") && fname.ends_with(".csv") {
                        let embed_name = fname
                            .strip_prefix("embeddings_")
                            .unwrap()
                            .strip_suffix(".csv")
                            .unwrap()
                            .to_string();

                        println!("found embedding file for model {}", embed_name);
                        let mut reader = csv::Reader::from_path(entry.path())?;

                        for result in reader.records() {
                            let record = result?;
                            let idx: usize = record.get(0).unwrap().parse().unwrap();
                            let mut vec = vec![];

                            for i in 1..record.len() {
                                let val: f32 = record.get(i).unwrap().parse().unwrap();
                                vec.push(val);
                            }

                            if let Some(prog) = res.programs.get_mut(idx) {
                                prog.set_embedding(embed_name.clone(), vec);
                            }
                        }
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }

        Ok(res)
    }

    pub fn do_experiments(&mut self, input: &GenerateInput) -> Result<(), LangExplorerError> {
        let len = self.programs.len();

        println!("computing indices");

        let indices: Vec<(usize, usize)> = (0..len)
            .flat_map(|i| ((i + 1)..len).map(move |j| (i, j)))
            .collect();

        println!("computing ast similarity scores");

        // For all experiments, we need to compute pairwise similarities between all ASTs.
        let ast_similarity_scores: Vec<f32> = indices
            .par_iter()
            .map(
                |(i, j)| match (self.programs.get(*i), self.programs.get(*j)) {
                    (Some(p1), Some(p2)) => wl_test(
                        &p1.features.as_slice(),
                        &p2.features.as_slice(),
                        VectorSimilarity::Euclidean,
                    ),
                    _ => panic!("we should have features here"),
                },
            )
            .collect();

        let mut ast_nn = self.compute_nn(&ast_similarity_scores, 5);
        for prog in self.programs.iter_mut() {
            prog.set_ast_nn(ast_nn.pop_front().unwrap());
        }

        let ast_distribution =
            Distribution::from_sample("ast_distribution", ast_similarity_scores.as_slice());

        let normalized_ast_scores = ast_distribution.minmax_scale(ast_similarity_scores.clone());

        let mut emb_c = vec![];
        let mut emb_n = vec![];
        let mut emb_d = vec![];
        let mut emb_nn = vec![];

        // Now, go through all embeddings and compute the similarity matrix, among other things.
        for emb in input.return_embeddings.iter() {
            println!("computing embedding similarity scores for {}", emb);
            let s = emb.to_string();

            let emb_similarity_scores: Vec<f32> = indices
                .par_iter()
                .map(
                    |(i, j)| match (self.programs.get(*i), self.programs.get(*j)) {
                        (Some(v1), Some(v2)) => {
                            match (v1.embeddings.get(&s), v2.embeddings.get(&s)) {
                                (Some(vec1), Some(vec2)) => {
                                    vector_similarity(vec1, vec2, VectorSimilarity::Euclidean)
                                }
                                _ => INFINITY,
                            }
                        }
                        _ => panic!("we should have features here"),
                    },
                )
                .collect();

            let emb_dist = Distribution::from_sample(
                emb.to_string().as_str(),
                emb_similarity_scores.as_slice(),
            );

            let normalized_emb_scores = emb_dist.minmax_scale(emb_similarity_scores.clone());

            let mut nn = self.compute_nn(&emb_similarity_scores, 5);
            for prog in self.programs.iter_mut() {
                prog.set_embedding_nn(emb.to_string(), nn.pop_front().unwrap());
            }

            emb_d.push(emb_dist);
            emb_c.push(emb_similarity_scores);
            emb_n.push(normalized_emb_scores);
            emb_nn.push(nn);
        }

        let similarity_results = self.compute_similarity_results(
            &emb_c,
            &emb_n,
            &ast_similarity_scores,
            &normalized_ast_scores,
            3,
            1.0,
        );

        self.similarity_experiments = Some(ExperimentResult {
            ast_distribution,
            embedding_distributions: emb_d,
            similarity_results,
        });

        Ok(())
    }

    fn compute_nn(&self, similarity_scores: &Vec<f32>, topk_len: usize) -> VecDeque<Vec<u32>> {
        let len = self.programs.len();

        // Get the nearest neighbors for each program.
        self.programs
            .par_iter()
            .enumerate()
            .map(|(idx, _)| {
                let mut topk: Vec<(f32, u32)> = vec![];

                let itertop = (0..idx)
                    .map(|i| (i, idx))
                    .chain(((idx + 1)..len).map(|j| (idx, j)));

                for (i, j) in itertop {
                    let idxfinal = (i * len) - ((i * (i + 1)) / 2) + (j - i - 1);
                    let other = similarity_scores[idxfinal];
                    let otheridx = if i == idx { j } else { i } as u32;
                    let elem = (other, otheridx);

                    let mut inserted = false;
                    for (pos, (val, _)) in topk.iter().enumerate() {
                        if other < *val {
                            topk.insert(pos, elem);
                            inserted = true;
                            break;
                        }
                    }

                    if !inserted {
                        topk.push(elem);
                    }

                    if topk.len() > topk_len {
                        topk.pop();
                    }
                }

                topk.iter().map(|(_, idx)| *idx).collect::<Vec<u32>>()
            })
            .collect::<VecDeque<Vec<u32>>>()
    }

    fn compute_similarity_results(
        &self,
        embedding_scores: &Vec<Vec<f32>>,
        normalized_embedding_similarity_scores: &Vec<Vec<f32>>,
        ast_scores: &Vec<f32>,
        normalized_ast_scores: &Vec<f32>,
        k: i32,
        gamma: f32,
    ) -> Vec<(f64, f64, f64, f64, f64)> {
        let mut similarity_results = vec![];
        let len = self.programs.len() as f64;

        for (embsim, normembsim) in embedding_scores
            .iter()
            .zip(normalized_embedding_similarity_scores.iter())
        {
            let mut avg_total: f64 = 0.0;
            let mut wavg_total: f64 = 0.0;
            let mut chisq_total: f64 = 0.0;
            let mut norm_avg_total: f64 = 0.0;
            let mut norm_chisq_total: f64 = 0.0;

            let simple_zip = embsim.iter().zip(ast_scores.iter());
            let norm_zip = normembsim.iter().zip(normalized_ast_scores.iter());

            for ((this, other), (norm_this, norm_other)) in simple_zip.zip(norm_zip) {
                let this = *this;
                let other = *other;
                let norm_this = *norm_this;
                let norm_other = *norm_other;

                let diff = (this - other).abs();
                let norm_diff = (norm_this - norm_other).abs();
                let sum = (this + other).abs();
                let chisq =
                    (diff.powi((k / 2) - 1) * E.powf(-diff / 2.0)) / (2.0_f32.powi(k / 2) * gamma);
                let norm_chisq = (norm_diff.powi((k / 2) - 1) * E.powf(-norm_diff / 2.0))
                    / (2.0_f32.powi(k / 2) * gamma);

                avg_total += diff as f64;
                wavg_total += ((2.0 * diff) / sum) as f64;
                chisq_total += chisq as f64;
                norm_avg_total += norm_diff as f64;
                norm_chisq_total += norm_chisq as f64;
            }

            similarity_results.push((
                avg_total / len,
                wavg_total / len,
                chisq_total / len,
                norm_avg_total / len,
                norm_chisq_total / len,
            ));
        }

        similarity_results
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

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct ExperimentResult {
    /// Distribution for AST similarities.
    ast_distribution: Distribution,
    /// Distributions for embedding similarities.
    embedding_distributions: Vec<Distribution>,
    /// Average, weighted, and chi-square similarity results, normalized average, normalized chi square.
    similarity_results: Vec<(f64, f64, f64, f64, f64)>,
}

impl GenerateOutput {
    async fn create_embeddings<B: Backend>(
        &mut self,
        embedding: EmbeddingModel,
        grammar: &Grammar,
        params: GeneralEmbeddingTrainingParams,
        models_dir: String,
        ollama_host: String,
        d2v_host: String,
        concurrent_requests: u64,
        tcp_keepalive_secs: u32,
    ) -> Result<(), LangExplorerError> {
        let emb_name = embedding.to_string();

        match &embedding {
            EmbeddingModel::Doc2VecDBOW => {
                // Hack for now, recompute the number of words, even
                // though this is going to be done again.
                let mut set = HashSet::new();
                let mut documents = vec![];

                for prog in self.programs.iter() {
                    documents.push(prog.program_internal.clone().unwrap());
                    for word in prog.features.iter() {
                        set.insert(*word);
                    }
                }

                let dim = params.d_model;
                let epochs = params.get_num_epochs();
                let params = Doc2VecDBOWNSEmbedderParams::new(
                    AdamWConfig::new(),
                    set.len(),
                    self.programs.len(),
                    params,
                    models_dir,
                );

                println!(
                            "training embeddings, there are {} documents being learned and {} total words being used",
                            documents.len(),
                            set.len(),
                        );

                let start = SystemTime::now();

                let model: Doc2VecEmbedderDBOWNS<Autodiff<B>> =
                    Doc2VecEmbedderDBOWNS::<Autodiff<B>>::new(grammar, params, Default::default())
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
                for prog in self.programs.iter_mut() {
                    prog.set_embedding(emb_name.clone(), embeddings.drain(0..dim).collect());
                }
            }
            EmbeddingModel::Doc2VecGensim => {
                let client = ClientBuilder::new()
                    .pool_max_idle_per_host(10)
                    .build()
                    .unwrap();

                let mut docs = HashMap::new();

                for prog in self.programs.iter() {
                    docs.insert(prog.program.clone().unwrap(), prog.features.clone());
                }

                match d2v::get_embedding_d2v(&client, &d2v_host, docs, &params.clone().into()).await
                {
                    Ok(resp) => {
                        for prog in self.programs.iter_mut() {
                            let vec = resp.get(&prog.program.clone().unwrap()).unwrap();
                            prog.set_embedding(emb_name.clone(), vec.clone());
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
            EmbeddingModel::MXBAILarge
            | EmbeddingModel::NomicEmbed
            | EmbeddingModel::SnowflakeArctic
            | EmbeddingModel::SnowflakeArctic2
            | EmbeddingModel::SnowflakeArctic137 => {
                let client = ClientBuilder::new()
                    .pool_max_idle_per_host(50)
                    .tcp_keepalive(Duration::from_secs(tcp_keepalive_secs as u64))
                    .build()
                    .unwrap();

                let blank: String = "".into();

                let programs: Vec<String> = self
                    .programs
                    .iter()
                    .map(|item| match &item.program {
                        Some(prompt) => prompt.clone(),
                        None => blank.clone(),
                    })
                    .collect();

                match get_embeddings_bulk_ollama(
                    &client,
                    &ollama_host,
                    programs.as_slice(),
                    embedding.clone(),
                    concurrent_requests as usize,
                )
                .await
                {
                    Ok(mut responses) => {
                        for (idx, vec) in responses.iter_mut().enumerate() {
                            let p = self.programs.get_mut(idx).unwrap();
                            let new = mem::take(vec);
                            p.set_embedding(emb_name.clone(), new);
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
        };

        Ok(())
    }

    pub fn write<P: Display>(
        &self,
        path: P,
        experiment_id: Option<usize>,
    ) -> Result<(), LangExplorerError> {
        let exp_id = match experiment_id {
            Some(id) => id,
            None => Self::get_experiment_id(&path, &self.language)?,
        };

        // Fix, need to create directories here too
        fs::create_dir_all(format!("{path}/{}/{exp_id}", self.language))?;

        let mut program_writer =
            csv::Writer::from_path(format!("{path}/{}/{exp_id}/programs.csv", self.language))?;

        let mut ast_nn_writer =
            csv::Writer::from_path(format!("{path}/{}/{exp_id}/ast_nn.csv", self.language))?;

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

            if idx == 0 {
                let mut vec = vec!["idx".to_string()];
                for i in 1..=prog.ast_nn.len() {
                    vec.push(format!("nn_{}", i));
                }

                ast_nn_writer.write_record(vec)?;
            }

            ast_nn_writer.write_field(idx.to_string())?;

            for val in prog.ast_nn.iter() {
                ast_nn_writer.write_field(val.to_string())?;
            }

            ast_nn_writer.write_record(None::<&[u8]>)?;

            // Flush every once in a while.
            if idx % 1000 == 0 {
                program_writer.flush()?;
                ast_nn_writer.flush()?;
            }
        }

        program_writer.flush()?;
        ast_nn_writer.flush()?;

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

            let mut embed_nn_writer = csv::Writer::from_path(format!(
                "{path}/{}/{exp_id}/embeddings_{}_nn.csv",
                self.language, embed_model
            ))?;

            for (idx, prog) in self.programs.iter().enumerate() {
                if let Some(emb) = prog.embeddings.get(&embed_model.to_string()) {
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

                if let Some(nn) = prog.embeddings_nn.get(&embed_model.to_string()) {
                    if idx == 0 {
                        let mut vec = vec!["idx".to_string()];
                        for i in 1..=nn.len() {
                            vec.push(format!("nn_{}", i));
                        }

                        embed_nn_writer.write_record(vec)?;
                    }

                    embed_nn_writer.write_field(idx.to_string())?;

                    for val in nn.iter() {
                        embed_nn_writer.write_field(val.to_string())?;
                    }

                    embed_nn_writer.write_record(None::<&[u8]>)?;
                }

                // Flush every once in a while.
                if idx % 1000 == 0 {
                    embed_writer.flush()?;
                    embed_nn_writer.flush()?;
                }
            }

            embed_writer.flush()?;
            embed_nn_writer.flush()?;

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

        self.write_experiments(&path, exp_id)?;

        Ok(())
    }

    /// Write out experiment results if any were run. This is refactored from the
    /// main write method so I can save experiment results separately.
    pub fn write_experiments<P: Display>(
        &self,
        path: P,
        exp_id: usize,
    ) -> Result<(), LangExplorerError> {
        if let Some(experiments) = &self.similarity_experiments {
            fs::write(
                format!("{path}/{}/{exp_id}/experiments.json", self.language),
                serde_json::to_string_pretty(&experiments)?,
            )?;
        }

        Ok(())
    }

    fn create_tsne(&self, embedding_name: String, output_path: String, dim: u8) {
        let mut vecs = vec![];

        for prog in self.programs.iter() {
            let emb = prog.embeddings.get(&embedding_name).unwrap();
            vecs.push(emb.as_slice());
        }

        match tSNE::new(&vecs)
            .embedding_dim(dim)
            .exact(|v1, v2| {
                let lsum = v1.iter().zip(v2.iter()).fold(0.0, |sum, (x1, x2)| {
                    let diff = x1 - x2;
                    sum + (diff * diff)
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
    #[serde(rename = "program")]
    program: Option<String>,

    /// Internal representation of the generated program.
    #[serde(skip_serializing, skip_deserializing)]
    program_internal: Option<ProgramInstance>,

    /// Optional graphviz representation for the generated program.
    #[serde(rename = "graphviz")]
    graphviz: Option<String>,

    /// If enabled, returns a list of all features extracted from
    /// the program.
    #[serde(rename = "features")]
    features: Vec<Feature>,

    #[serde(rename = "ast_nn")]
    /// If enabled, returns the nearest neighbors in the dataset of each AST.
    ast_nn: Vec<u32>,

    /// If enabled, returns the embedding of the program.
    #[serde(rename = "embeddings")]
    embeddings: HashMap<String, Vec<f32>>,

    /// If enabled, returns nearest-neighbor embeddings for each embedding vector.
    #[serde(rename = "embeddings_nn")]
    embeddings_nn: HashMap<String, Vec<u32>>,

    /// If enabled, returns t-SNE embeddings for each embedding vector.
    #[serde(rename = "tsne_2d")]
    tnse_2d: Option<HashMap<String, Vec<f32>>>,

    /// If enabled, returns the program graph in edge-list format.
    #[serde(rename = "edge_list")]
    edge_list: Option<Vec<(InstanceId, InstanceId)>>,

    /// Toggle whether or not the generated program is a partial program or not.
    #[serde(rename = "is_partial")]
    is_partial: bool,
}

impl ProgramResult {
    pub(crate) fn new() -> Self {
        Self {
            program: None,
            program_internal: None,
            graphviz: None,
            features: vec![],
            ast_nn: vec![],
            embeddings: HashMap::new(),
            embeddings_nn: HashMap::new(),
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
        self.features = features;
    }

    pub(crate) fn set_edge_list(&mut self, edge_list: Vec<(InstanceId, InstanceId)>) {
        self.edge_list = Some(edge_list);
    }

    pub(crate) fn set_embedding(&mut self, name: String, embedding: Vec<f32>) {
        self.embeddings.insert(name, embedding);
    }

    pub(crate) fn set_embedding_nn(&mut self, name: String, nn: Vec<u32>) {
        self.embeddings_nn.insert(name, nn);
    }

    pub(crate) fn set_ast_nn(&mut self, nn: Vec<u32>) {
        self.ast_nn = nn;
    }
}

impl From<ProgramRecord> for ProgramResult {
    fn from(value: ProgramRecord) -> Self {
        ProgramResult {
            program: Some(value.program),
            program_internal: None,
            graphviz: None,
            features: vec![],
            ast_nn: vec![],
            embeddings: HashMap::new(),
            embeddings_nn: HashMap::new(),
            tnse_2d: None,
            edge_list: None,
            is_partial: value.is_partial,
        }
    }
}
