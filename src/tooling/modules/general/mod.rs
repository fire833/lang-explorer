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
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::{
        activation::{leaky_relu, relu, sigmoid, tanh},
        Float, Tensor,
    },
};

use crate::tooling::modules::expander::Activation;

#[derive(Debug, Config)]
pub struct GeneralLinearConfig {
    /// The sizes of each layer in the model.
    pub layers: Vec<usize>,

    /// Optionally toggle bias within the model.
    #[config(default = true)]
    pub bias: bool,
}

impl GeneralLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GeneralLinear<B> {
        let mut layers = vec![];

        for item in self.layers.windows(2) {
            let layer = LinearConfig::new(item[0], item[1])
                .with_bias(self.bias)
                .init(device);
            layers.push(layer);
        }

        GeneralLinear { layers }
    }
}

#[derive(Debug, Module)]
pub struct GeneralLinear<B: Backend> {
    layers: Vec<Linear<B>>,
}

impl<B: Backend> GeneralLinear<B> {
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<B, D, Float>,
        activation: Activation,
    ) -> Tensor<B, D, Float> {
        let mut x = input;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);

            // Apply activation function to all layers except the last
            if i < self.layers.len() - 1 {
                x = match activation {
                    Activation::ReLU => relu(x),
                    Activation::LeakyReLU(slope) => leaky_relu(x, (1.0 / 10000.0) * slope as f64),
                    Activation::Sigmoid => sigmoid(x),
                    Activation::TanH => tanh(x),
                };
            }
        }

        x
    }
}

#[test]
fn test_forward() {}
