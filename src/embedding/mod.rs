/*
*	Copyright (C) 2026 Kendall Tauser
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

use std::collections::{BTreeMap, HashSet};
use std::fmt::Display;
use std::hash::Hash;
use std::path::Path;
use std::str::FromStr;

use burn::record::{HalfPrecisionSettings, PrettyJsonFileRecorder};
use burn::{
    config::Config,
    tensor::{backend::AutodiffBackend, Device, Tensor},
};
use clap::ValueEnum;
use rand::Rng;
#[allow(unused)]
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    errors::LangExplorerError,
    grammar::grammar::Grammar,
    tooling::{modules::embed::AggregationMethod, training::TrainingParams},
};

pub mod doc2vecdbowns;
pub mod doc2vecdm;
pub mod graphmae;

/// Main trait for creating embeddings of programs.
pub trait LanguageEmbedder<B: AutodiffBackend> {
    type Document;
    type Params;

    /// Initializes an Embedder system. This typically involves either
    /// initializing a new model, or retrieving an already trained model
    /// instance from disk.
    fn new(grammar: &Grammar, params: Self::Params, device: Device<B>) -> Self;

    /// Trains the embedder on the provided corpus.
    fn fit(self, documents: &[Self::Document]) -> Result<Self, LangExplorerError>
    where
        Self: Sized;

    /// Creates an embedding given the current model on a new
    /// document and it's corresponding words.
    fn embed(&mut self, document: Self::Document) -> Result<Tensor<B, 1>, LangExplorerError>;

    /// Returns the embeddings of the documents that were trained on.
    fn get_embeddings(&self) -> Result<Vec<f32>, LangExplorerError>;
}

#[derive(Config, Debug, ToSchema)]
pub struct GeneralEmbeddingTrainingParams {
    /// The dimension of embeddings within the model.
    #[config(default = 128)]
    pub d_model: usize,
    /// The number of words to the left of the center word
    /// to predict on.
    #[config(default = 5)]
    pub window_left: usize,
    /// The number of words to the right of the center word
    /// to predict on.
    #[config(default = 5)]
    pub window_right: usize,
    /// The number of negative samples to update if using the
    /// negative sampling loss function.
    #[config(default = 4)]
    pub num_neg_samples: usize,
    /// The aggregation method to use.
    #[config(default = "AggregationMethod::Average")]
    pub agg: AggregationMethod,
    /// General training params.
    pub gen_params: TrainingParams,
}

impl Default for GeneralEmbeddingTrainingParams {
    fn default() -> Self {
        Self {
            d_model: 128,
            window_left: 5,
            window_right: 5,
            num_neg_samples: 5,
            agg: AggregationMethod::Average,
            gen_params: TrainingParams::default(),
        }
    }
}

impl GeneralEmbeddingTrainingParams {
    pub fn get_batch_size(&self) -> usize {
        self.gen_params.batch_size
    }

    pub fn get_num_epochs(&self) -> usize {
        self.gen_params.n_epochs
    }

    pub fn get_learning_rate(&self) -> f64 {
        self.gen_params.learning_rate
    }

    pub fn get_seed(&self) -> u64 {
        self.gen_params.seed
    }

    pub fn get_display_frequency(&self) -> usize {
        self.gen_params.display_frequency
    }

    pub fn get_save_model(&self) -> bool {
        self.gen_params.save_model
    }

    pub fn get_create_new_model(&self) -> bool {
        self.gen_params.create_new_model
    }

    pub fn get_model_recorder(&self) -> PrettyJsonFileRecorder<HalfPrecisionSettings> {
        PrettyJsonFileRecorder::<HalfPrecisionSettings>::new()
    }
}

pub(super) fn get_positive_indices<R: Rng, W: Ord>(
    wordset: &BTreeMap<W, u32>,
    doc_words: &[W],
    num_positive_samples: usize,
    rng: &mut R,
) -> Vec<usize> {
    if doc_words.len() <= num_positive_samples {
        return doc_words
            .iter()
            .map(|w| *wordset.get(w).unwrap() as usize)
            .collect::<Vec<usize>>();
    }

    let mut positive_samples = HashSet::new();
    while positive_samples.len() < num_positive_samples {
        let word = doc_words
            .get(rng.random::<u32>() as usize % doc_words.len())
            .unwrap();
        if let Some(idx) = wordset.get(word) {
            let idx = *idx as usize;
            if !positive_samples.contains(&idx) {
                positive_samples.insert(idx);
            }
        }
    }

    positive_samples.into_iter().collect()
}

#[test]
fn test_positive_indices() {
    use rand_chacha::ChaCha8Rng;

    let mut set: BTreeMap<usize, u32> = BTreeMap::new();
    for i in 0..1000 {
        set.insert(i, i as u32);
    }

    let mut rng = ChaCha8Rng::seed_from_u64(10);

    let indices = get_positive_indices(
        &set,
        &vec![10, 50, 555, 556, 557, 668, 537, 23, 234, 343, 129],
        1,
        &mut rng,
    );
    println!("indices: {indices:?}");
}

pub(super) fn get_negative_indices<R: Rng, W: Ord + Hash>(
    wordset: &BTreeMap<u32, W>,
    doc_words: &HashSet<W>,
    num_negative_samples: usize,
    rng: &mut R,
) -> Vec<usize> {
    let mut negative_samples = HashSet::new();
    while negative_samples.len() < num_negative_samples {
        let idx = rng.random::<u32>() % wordset.len() as u32;
        let word = wordset.get(&idx).unwrap();
        if !doc_words.contains(word) && !negative_samples.contains(&(idx as usize)) {
            negative_samples.insert(idx as usize);
        }
    }

    negative_samples.into_iter().collect()
}

#[test]
fn test_negative_indices() {
    use rand_chacha::ChaCha8Rng;

    let mut set: BTreeMap<u32, usize> = BTreeMap::new();
    for i in 0..1000 {
        set.insert(i as u32, i);
    }

    let mut rng = ChaCha8Rng::seed_from_u64(10);

    let hash: HashSet<usize> = vec![10, 50, 555, 556, 557, 668, 537, 23, 234, 343, 129]
        .into_iter()
        .collect();

    let indices = get_negative_indices(&set, &hash, 10, &mut rng);
    println!("indices: {indices:?}");
}

pub(super) fn save_embeddings_as_csv<P: AsRef<Path>>(
    embeddings: &Vec<f32>,
    dim: usize,
    path: P,
) -> Result<(), LangExplorerError> {
    let num = embeddings.len() / dim;
    let mut writer = csv::Writer::from_path(path)?;
    let emb = embeddings.as_slice();

    for n in 0..num {
        let slice = &emb[n * dim..(n + 1) * dim];
        writer.serialize(slice)?;
    }

    Ok(())
}

#[derive(Debug, Clone, ValueEnum, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingModel {
    #[serde(rename = "doc2vecdbow")]
    Doc2VecDBOW,
    #[serde(rename = "doc2vecgensim")]
    Doc2VecGensim,
    #[serde(rename = "mxbai-embed-large")]
    MXBAILarge,
    #[serde(rename = "nomic-embed-text")]
    NomicEmbed,
    #[serde(rename = "snowflake-arctic-embed:137m")]
    SnowflakeArctic137,
    #[serde(rename = "snowflake-arctic-embed")]
    SnowflakeArctic,
    #[serde(rename = "snowflake-arctic-embed2")]
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
