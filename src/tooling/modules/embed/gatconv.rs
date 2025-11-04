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
    nn::{Dropout, DropoutConfig},
    prelude::Backend,
    tensor::{Float, Tensor},
};

use crate::tooling::modules::{
    expander::Activation,
    general::{GeneralLinear, GeneralLinearConfig},
};

#[derive(Debug, Config)]
pub struct GATConvConfig {
    /// Configuration for the fully connected layer.
    /// An additional layer will be added to the front of size d_in.
    pub fc_layers: Vec<usize>,

    /// Dropout rate for the dropout modules.
    pub dropout_rate: f32,

    /// Dimension of node input features (h).
    pub d_in: usize,
    /// Dimension of node output features (h').
    pub d_out: usize,
}

impl GATConvConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GATConv<B> {
        let mut layers = self.fc_layers.clone();
        layers.insert(0, self.d_in);

        GATConv {
            linear: GeneralLinearConfig::new(layers)
                .with_bias(false)
                .init(device),
            attn_l: GeneralLinearConfig::new(vec![self.d_in, 1])
                .with_bias(false)
                .init(device),
            attn_r: GeneralLinearConfig::new(vec![self.d_in, 1])
                .with_bias(false)
                .init(device),
            feat_drop: DropoutConfig::new(self.dropout_rate as f64).init(),
            attn_drop: DropoutConfig::new(self.dropout_rate as f64).init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct GATConv<B: Backend> {
    linear: GeneralLinear<B>,

    attn_l: GeneralLinear<B>,
    attn_r: GeneralLinear<B>,

    feat_drop: Dropout,
    attn_drop: Dropout,
}

impl<B: Backend> GATConv<B> {
    pub fn forward(
        &self,
        node_features: Tensor<B, 3, Float>,
        activation: Activation,
    ) -> Tensor<B, 3, Float> {
        todo!()
    }
}

#[test]
fn test_forward() {}
