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
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3, Float> {
        let vecs = self.embed.forward(input);
        let out = self.hidden.forward(vecs);
        return out;
    }
}
