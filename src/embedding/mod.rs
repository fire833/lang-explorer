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

use burn::{
    config::Config,
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Device, Tensor},
};

use crate::{
    errors::LangExplorerError,
    grammar::{grammar::Grammar, NonTerminal, Terminal},
    tooling::{modules::embed::AggregationMethod, training::TrainingParams},
};

pub mod doc2vecdbow;
pub mod doc2vecdm;

/// Main trait for creating embeddings of programs.
pub trait LanguageEmbedder<T: Terminal, I: NonTerminal, B: AutodiffBackend> {
    type Document;
    type Word: PartialEq + PartialOrd;
    type Params;

    /// Initializes an Embedder system. This typically involves either
    /// initializing a new model, or retrieving an already trained model
    /// instance from disk.
    fn new(grammar: &Grammar<T, I>, params: Self::Params, device: Device<B>) -> Self;

    /// Trains the embedder on the provided corpus.
    fn fit(
        self,
        documents: &[(Self::Document, Vec<Self::Word>)],
    ) -> Result<Self, LangExplorerError>
    where
        Self: Sized;

    /// Creates an embedding given the current model on a new
    /// document and it's corresponding words.
    fn embed(
        &mut self,
        document: (Self::Document, Vec<Self::Word>),
    ) -> Result<Tensor<B, 1>, LangExplorerError>;

    /// Returns the embeddings of the documents that were trained on.
    fn get_embeddings(&self) -> Result<Vec<f64>, LangExplorerError>;
}

#[derive(Config, Debug)]
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
    #[config(default = 32)]
    pub n_neg_samples: usize,
    /// The aggregation method to use.
    pub agg: AggregationMethod,
    /// General training params.
    pub gen_params: TrainingParams,
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
}
