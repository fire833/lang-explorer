/*
*	Copyright (C) 2026 Kendall Tauser
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

use crate::{grammar::program::ProgramInstance, tooling::modules::embed::gcnconv::GCNConv};

#[derive(Debug, Config)]
pub struct GraphConvolutionNetConfig {}

impl GraphConvolutionNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GraphConvolutionNet<B> {
        GraphConvolutionNet { layers: vec![] }
    }
}

#[derive(Debug, Module)]
pub struct GraphConvolutionNet<B: Backend> {
    layers: Vec<GCNConv<B>>,
}

impl<B: Backend> GraphConvolutionNet<B> {
    pub fn forward(
        &self,
        mut node_features: Tensor<B, 3, Float>,
        programs: &Vec<&ProgramInstance>,
    ) -> Tensor<B, 3, Float> {
        for layer in self.layers.iter() {
            node_features = layer.forward(node_features, programs);
        }

        node_features
    }
}

#[test]
fn test_forward() {}
