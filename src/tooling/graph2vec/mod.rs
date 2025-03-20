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
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Tensor},
};

pub struct Graph2Vec<B: Backend> {
    embed: Embedding<B>,
}

#[derive(Debug, Config)]
pub struct Graph2VecConfig {
    embedding_dim: u64,
    num_graphs: u64,
}

impl Graph2VecConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Graph2Vec<B> {
        Graph2Vec {
            embed: EmbeddingConfig::new(self.num_graphs as usize, self.embedding_dim as usize)
                .init(device),
        }
    }
}

impl<B: Backend> Graph2Vec<B> {
    pub fn forward(&self, graphs: Tensor<B, 2>) -> Tensor<B, 3> {
        let [dim, num_samples] = graphs.dims();
        todo!()
        // let x = graphs.reshape([]);
        // self.embed.forward(graphs)
    }
}
