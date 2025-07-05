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
    tensor::{backend::Backend, Tensor},
};

use crate::tooling::modules::embed::pvdbow::Doc2VecDBOW;

#[derive(Debug, Config)]
pub struct Graph2VecConfig {
    /// The number of
    pub n_graphs: usize,
    pub d_model: usize,
}

impl Graph2VecConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Graph2Vec<B> {
        todo!()
    }
}

#[derive(Debug, Module)]
pub struct Graph2Vec<B: Backend> {
    doc: Doc2VecDBOW<B>,
}

impl<B: Backend> Graph2Vec<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 3> {
        let [_dim, _num_samples] = input.dims();
        todo!()
        // let x = graphs.reshape([]);
        // self.embed.forward(graphs)
    }
}
