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
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Float, Int, Tensor},
};

#[derive(Debug, Config)]
pub struct Doc2VecDMConfig {
    /// The number of word vectors.
    pub n_words: usize,
    /// The number of document vectors.
    pub n_docs: usize,
    /// The size of each vector.
    pub d_model: usize,
}

impl Doc2VecDMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Doc2VecDM<B> {
        Doc2VecDM {
            words: EmbeddingConfig::new(self.n_words, self.d_model).init(device),
            documents: EmbeddingConfig::new(self.n_docs, self.d_model).init(device),
            hidden: LinearConfig::new(self.d_model, self.n_words)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct Doc2VecDM<B: Backend> {
    /// Embeddings of all words within the corpus.
    words: Embedding<B>,
    /// Embeddings of all documents within the corpus.
    documents: Embedding<B>,
    /// The hidden layer to compute output logits.
    hidden: Linear<B>,
}

impl<B: Backend> Doc2VecDM<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2, Float> {
        todo!()
    }
}
