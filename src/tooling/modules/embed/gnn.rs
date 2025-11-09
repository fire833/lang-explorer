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

use crate::tooling::modules::{
    embed::{
        gat::{GraphAttentionNet, GraphAttentionNetConfig},
        gcn::{GraphConvolutionNet, GraphConvolutionNetConfig},
    },
    expander::Activation,
};

#[derive(Debug, Config)]
pub enum GraphNeuralNetConfig {
    GCN(GraphConvolutionNetConfig),
    GAT(GraphAttentionNetConfig),
}

impl GraphNeuralNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GraphNeuralNet<B> {
        match self {
            GraphNeuralNetConfig::GCN(config) => GraphNeuralNet::GCN(config.init(device)),
            GraphNeuralNetConfig::GAT(config) => GraphNeuralNet::GAT(config.init(device)),
        }
    }
}

#[derive(Debug, Module)]
pub enum GraphNeuralNet<B: Backend> {
    GCN(GraphConvolutionNet<B>),
    GAT(GraphAttentionNet<B>),
}

impl<B: Backend> GraphNeuralNet<B> {
    pub fn forward(&self, node_features: Tensor<B, 3, Float>) -> Tensor<B, 3, Float> {
        match self {
            GraphNeuralNet::GCN(gcn) => gcn.forward(node_features),
            GraphNeuralNet::GAT(gat) => gat.forward(node_features, Activation::LeakyReLU(20)),
        }
    }
}

#[test]
fn test_forward() {}
