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

// Notes: https://towardsdatascience.com/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0/
// https://github.com/OlgaChernytska/word2vec-pytorch/blob/87b0418fcc6a0f5b8ac96698f6fc1079014b4615/utils/trainer.py
// https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py
// https://github.com/mklf/word2vec-rs
// https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html

use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Float, Int, Tensor},
};

#[derive(Debug, Config)]
pub struct Word2VecSGConfig {
    /// The number of embedding vectors.
    pub n_words: usize,
    /// The size of each vector.
    pub d_model: usize,
}

impl Word2VecSGConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Word2VecSG<B> {
        Word2VecSG {
            embed: EmbeddingConfig::new(self.n_words, self.d_model).init(device),
            hidden: LinearConfig::new(self.d_model, self.n_words)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct Word2VecSG<B: Backend> {
    /// The input embedding.
    embed: Embedding<B>,
    /// The hidden layer to compute output logits.
    hidden: Linear<B>,
}

impl<B: Backend> Word2VecSG<B> {
    /// Applies the forward pass to the input tensor.
    ///
    /// More specifically, the forward pass expects an array of indices
    /// of middle words, and returns an array of the logits for each `batch_item`
    /// of size `n_words` corresponding to the probability of each word
    /// being the context word.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size]`
    /// - output: `[batch_size, n_words]`
    pub fn forward(&self, input: Tensor<B, 1, Int>) -> Tensor<B, 2, Float> {
        let vecs = self.embed.forward(input.unsqueeze_dim::<2>(0));
        let out = self.hidden.forward(vecs);
        return out.squeeze::<2>(0);
    }
}

#[test]
fn test_forward() {}
