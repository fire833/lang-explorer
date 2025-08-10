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
        activation::{relu, sigmoid},
        Float, Tensor,
    },
};

use crate::tooling::modules::expander::Activation;

#[derive(Debug, Config)]
pub struct Linear2DeepConfig {
    /// The size of the input embeddings.
    #[config(default = 128)]
    pub d_embed: usize,
    /// The size of the hidden layer.
    #[config(default = 64)]
    pub d_hidden: usize,
    /// The number of output productions rules
    pub n_productions: usize,
    /// Optionally toggle bias within the model.
    #[config(default = true)]
    pub bias: bool,
}

impl Linear2DeepConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Linear2Deep<B> {
        Linear2Deep {
            input: LinearConfig::new(self.d_embed, self.d_hidden)
                .with_bias(self.bias)
                .init(device),
            output: LinearConfig::new(self.d_hidden, self.n_productions)
                .with_bias(self.bias)
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct Linear2Deep<B: Backend> {
    input: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> Linear2Deep<B> {
    pub fn forward(
        &self,
        embed: Tensor<B, 2, Float>,
        activation: Activation,
    ) -> Tensor<B, 2, Float> {
        let x = self.input.forward(embed);
        let x = match activation {
            Activation::Sigmoid => sigmoid(x),
            Activation::ReLU => relu(x),
        };

        self.output.forward(x)
    }
}

#[test]
fn test_forward() {}
