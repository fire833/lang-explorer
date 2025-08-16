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
    data::dataloader::batcher::Batcher,
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Device, Int, Tensor},
};
use tokio::sync::mpsc::{channel, Receiver, Sender};

use crate::{
    errors::LangExplorerError,
    grammar::{grammar::Grammar, NonTerminal, Terminal},
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

/// A training item represents a single item in a batch that should be trained on.
#[derive(Debug, Clone)]
pub(super) struct ProgramTrainingItem {
    document_idx: usize,
    context_word_indices: Vec<usize>,
    center_word_idx: usize,
    negative_sample_indices: Vec<usize>,
}

impl ProgramTrainingItem {
    fn new(
        document_idx: usize,
        center_word_idx: usize,
        context_word_indices: Vec<usize>,
        negative_sample_indices: Vec<usize>,
    ) -> Self {
        Self {
            document_idx,
            context_word_indices,
            center_word_idx,
            negative_sample_indices,
        }
    }
}

/// A batch of document/word context items that are already in their correct tensor form.
pub(super) struct ProgramBatch<B: Backend> {
    documents: Tensor<B, 1, Int>,
    context_words: Tensor<B, 2, Int>,
    negative_words: Tensor<B, 2, Int>,
    true_words: Tensor<B, 1, Int>,
}

#[derive(Debug, Clone)]
pub(super) struct ProgramBatcher {}

impl ProgramBatcher {
    fn new() -> Self {
        Self {}
    }
}

impl<B: Backend> Batcher<B, ProgramTrainingItem, ProgramBatch<B>> for ProgramBatcher {
    fn batch(&self, items: Vec<ProgramTrainingItem>, device: &B::Device) -> ProgramBatch<B> {
        let mut doc_idx = vec![];
        let mut center_word_idx = vec![];
        let mut context_word_idx: Vec<Tensor<B, 1, Int>> = vec![];
        let mut negative_indices: Vec<Tensor<B, 1, Int>> = vec![];

        for item in items.iter() {
            doc_idx.push(item.document_idx as i32);
            center_word_idx.push(item.center_word_idx as i32);
            let ctx_word_indices: Vec<i32> = item
                .context_word_indices
                .iter()
                .map(|v| *v as i32)
                .collect();

            context_word_idx.push(Tensor::from_data(ctx_word_indices.as_slice(), device));
            negative_indices.push(Tensor::from_data(
                item.negative_sample_indices.as_slice(),
                device,
            ));
        }

        ProgramBatch {
            documents: Tensor::from_data(doc_idx.as_slice(), device),
            context_words: Tensor::stack::<2>(context_word_idx, 0),
            negative_words: Tensor::stack::<2>(negative_indices, 0),
            true_words: Tensor::from_data(center_word_idx.as_slice(), device),
        }
    }
}
