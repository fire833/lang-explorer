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
    nn::{Dropout, DropoutConfig, LeakyRelu, LeakyReluConfig},
    prelude::Backend,
    tensor::{Float, Tensor},
};

use crate::{
    grammar::program::ProgramInstance,
    tooling::modules::{
        expander::Activation,
        general::{GeneralLinear, GeneralLinearConfig},
    },
};

#[derive(Debug, Config)]
pub struct GATHeadConfig {
    /// Configuration for the fully connected layer.
    /// An additional layer will be added to the front of size d_in.
    pub fc_layers: Vec<usize>,

    /// Dropout rate for the dropout modules.
    pub dropout_rate: f32,

    /// Dimension of node input features (h).
    pub d_in: usize,
    /// Dimension of node output features (h').
    pub d_out: usize,

    /// Slope of the leaky relu.
    pub leaky_relu_slope: f32,
}

impl GATHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GATHead<B> {
        let mut layers = self.fc_layers.clone();
        layers.insert(0, self.d_in);
        let no_activations = vec![false; layers.len() - 1];

        GATHead {
            linear: GeneralLinearConfig::new(layers, no_activations)
                .with_bias(false)
                .init(device),
            attn_l: GeneralLinearConfig::new(vec![self.d_in, 1], vec![true])
                .with_bias(false)
                .init(device),
            attn_r: GeneralLinearConfig::new(vec![self.d_in, 1], vec![true])
                .with_bias(false)
                .init(device),
            feat_drop: DropoutConfig::new(self.dropout_rate as f64).init(),
            attn_drop: DropoutConfig::new(self.dropout_rate as f64).init(),
            leaky_relu: LeakyReluConfig::new()
                .with_negative_slope(self.leaky_relu_slope as f64)
                .init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct GATHead<B: Backend> {
    linear: GeneralLinear<B>,

    attn_l: GeneralLinear<B>,
    attn_r: GeneralLinear<B>,

    leaky_relu: LeakyRelu,

    feat_drop: Dropout,
    attn_drop: Dropout,
}

impl<B: Backend> GATHead<B> {
    pub fn forward(
        &self,
        node_features: Tensor<B, 3, Float>,
        programs: &Vec<&ProgramInstance>,
        activation: Activation,
    ) -> Tensor<B, 3, Float> {
        let n_dropout = self.feat_drop.forward(node_features);
        let w_xij = self.linear.forward(n_dropout, activation.clone());
        let att_l = self
            .leaky_relu
            .forward(self.attn_l.forward(w_xij.clone(), activation.clone()));
        let att_r = self
            .leaky_relu
            .forward(self.attn_r.forward(w_xij.clone(), activation.clone()));
        let att_l = self.attn_drop.forward(att_l);
        let att_r = self.attn_drop.forward(att_r);

        todo!()
    }
}

#[test]
fn test_forward() {}
