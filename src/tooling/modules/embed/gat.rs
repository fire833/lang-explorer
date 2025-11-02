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
    prelude::Backend,
    tensor::{Float, Tensor},
};

use crate::tooling::modules::embed::gatconv::GATConv;

#[derive(Debug, Config)]
pub struct GraphAttentionNetConfig {}

impl GraphAttentionNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GraphAttentionNet<B> {
        GraphAttentionNet { layers: vec![] }
    }
}

#[derive(Debug, Module)]
pub struct GraphAttentionNet<B: Backend> {
    layers: Vec<GATConv<B>>,
}

impl<B: Backend> GraphAttentionNet<B> {
    pub fn forward(&self, mut input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        for layer in self.layers.iter() {
            input = layer.forward();
        }

        input
    }
}

#[test]
fn test_forward() {}
