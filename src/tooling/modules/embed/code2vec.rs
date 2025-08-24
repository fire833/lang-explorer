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
    tensor::{Float, Tensor},
};

#[derive(Debug, Config)]
#[allow(unused)]
pub struct Code2VecConfig {
    /// The number of terminals.
    pub n_terminals: usize,
    /// The number of code paths to learn on.
    pub n_paths: usize,
    /// The number of output labels.
    pub n_labels: usize,
    /// The size of each terminal vector.
    pub d_terminal: usize,
    /// The size of each path vector.
    pub d_path: usize,
    /// The dimension of the encoded vectors after
    /// passing through the linear input.
    pub d_encode: usize,
}

impl Code2VecConfig {
    #[allow(unused)]
    pub fn init<B: Backend>(&self, device: &B::Device) -> Code2Vec<B> {
        Code2Vec {
            terminals: EmbeddingConfig::new(self.n_terminals, self.d_terminal).init(device),
            paths: EmbeddingConfig::new(self.n_paths, self.d_path).init(device),
            hidden_in: LinearConfig::new(2 * self.d_terminal + self.d_path, self.d_encode)
                .with_bias(false)
                .init(device),
            hidden_out: LinearConfig::new(self.d_encode, self.n_labels)
                .with_bias(true)
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct Code2Vec<B: Backend> {
    /// Embeddings of all terminals within the corpus.
    terminals: Embedding<B>,
    /// Embeddings of all paths within the corpus.
    paths: Embedding<B>,
    /// The hidden input layer.
    hidden_in: Linear<B>,
    /// The hidden output layer.
    hidden_out: Linear<B>,
}

impl<B: Backend> Code2Vec<B> {
    #[allow(unused)]
    pub fn forward(&self) -> Tensor<B, 2, Float> {
        todo!()
    }
}

#[test]
fn test_forward() {}
