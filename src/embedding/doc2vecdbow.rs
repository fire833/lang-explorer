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

use std::marker::PhantomData;

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    optim::{adaptor::OptimizerAdaptor, AdamW, AdamWConfig},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Device, Float, Int, Tensor},
    train::{TrainOutput, TrainStep},
};
use rand_chacha::ChaCha8Rng;

use crate::{
    embedding::{GeneralEmbeddingTrainingParams, LanguageEmbedder},
    errors::LangExplorerError,
    grammar::{grammar::Grammar, NonTerminal, Terminal},
    languages::Feature,
    tooling::modules::{embed::pvdbow::Doc2VecDBOW, loss::nsampling::NegativeSampling},
};

pub struct Doc2VecEmbedderDBOW<T: Terminal, I: NonTerminal, B: AutodiffBackend> {
    // Some bullcrap.
    d1: PhantomData<T>,
    d2: PhantomData<I>,

    model: Doc2VecDBOW<B>,
    loss: NegativeSampling<B>,

    device: Device<B>,

    /// RNG stuff.
    rng: ChaCha8Rng,

    optim: OptimizerAdaptor<AdamW, Doc2VecDBOW<B>, B>,

    params: GeneralEmbeddingTrainingParams,
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend>
    TrainStep<ProgramBatch<B>, Tensor<B, 1, Float>> for Doc2VecEmbedderDBOW<T, I, B>
{
    fn step(&self, item: ProgramBatch<B>) -> TrainOutput<Tensor<B, 1, Float>> {
        let logits = self.model.forward(item.documents);
        let loss = self
            .loss
            .forward(item.true_words, item.negative_words, logits);

        TrainOutput::new(&self.model, loss.backward(), loss)
    }
}

#[derive(Config)]
pub struct Doc2VecDBOWEmbedderParams {
    /// Configuration for Adam.
    pub ada_config: AdamWConfig,
    /// The number of words within the model.
    pub n_words: usize,
    /// The number of documents within the model.
    pub n_docs: usize,
    /// General parameters shared among all embedder params.
    pub gen_params: GeneralEmbeddingTrainingParams,
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> LanguageEmbedder<T, I, B>
    for Doc2VecEmbedderDBOW<T, I, B>
{
    type Document = String;
    type Word = Feature;
    type Params = Doc2VecDBOWEmbedderParams;

    fn new(grammar: &Grammar<T, I>, params: Self::Params, device: Device<B>) -> Self {
        todo!()
    }

    fn fit(self, documents: &[(Self::Document, Vec<Self::Word>)]) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        todo!()
    }

    fn embed(
        &mut self,
        document: (Self::Document, Vec<Self::Word>),
    ) -> Result<Tensor<B, 1>, LangExplorerError> {
        todo!()
    }

    fn get_embeddings(&self) -> Result<Vec<f64>, LangExplorerError> {
        todo!()
    }
}

/// A training item represents a single item in a batch that should be trained on.
#[derive(Debug, Clone)]
struct ProgramTrainingItem {
    document_idx: usize,
    true_word_indices: Vec<usize>,
    negative_sample_indices: Vec<usize>,
}

impl ProgramTrainingItem {
    fn new(
        document_idx: usize,
        true_word_indices: Vec<usize>,
        negative_sample_indices: Vec<usize>,
    ) -> Self {
        Self {
            document_idx,
            true_word_indices,
            negative_sample_indices,
        }
    }
}

/// A batch of document/word context items that are already in their correct tensor form.
struct ProgramBatch<B: Backend> {
    documents: Tensor<B, 1, Int>,
    negative_words: Tensor<B, 2, Int>,
    true_words: Tensor<B, 2, Int>,
}

#[derive(Debug, Clone)]
struct ProgramBatcher {}

impl ProgramBatcher {
    fn new() -> Self {
        Self {}
    }
}

impl<B: Backend> Batcher<B, ProgramTrainingItem, ProgramBatch<B>> for ProgramBatcher {
    fn batch(&self, items: Vec<ProgramTrainingItem>, device: &B::Device) -> ProgramBatch<B> {
        let mut doc_idx = vec![];
        let mut true_word_indices: Vec<Tensor<B, 1, Int>> = vec![];
        let mut negative_indices: Vec<Tensor<B, 1, Int>> = vec![];

        for item in items.iter() {
            doc_idx.push(item.document_idx as i32);
            true_word_indices.push(Tensor::from_data(item.true_word_indices.as_slice(), device));
            negative_indices.push(Tensor::from_data(
                item.negative_sample_indices.as_slice(),
                device,
            ));
        }

        ProgramBatch {
            documents: Tensor::from_data(doc_idx.as_slice(), device),
            negative_words: Tensor::stack::<2>(negative_indices, 0),
            true_words: Tensor::stack::<2>(true_word_indices, 0),
        }
    }
}
