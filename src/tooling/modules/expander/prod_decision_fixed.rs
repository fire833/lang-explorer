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
    nn::{Embedding, EmbeddingConfig},
    prelude::Backend,
    tensor::{Float, Int, Tensor},
};

use crate::tooling::modules::expander::{
    lin2::{Linear2Deep, Linear2DeepConfig},
    Activation,
};

#[derive(Debug, Config)]
pub struct ProductionDecisionFixedConfig {
    /// The number of dimensions of embeddings.
    d_model: usize,

    /// The dimension of input vectors.
    d_embed: usize,

    /// The number of embeddings to store. This will typically
    /// correspond to the number of rules within the grammar, since
    /// we will want to score the best rule to expand.
    n_embed: usize,
}

impl ProductionDecisionFixedConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ProductionDecisionFixed<B> {
        ProductionDecisionFixed {
            rule_embeddings: EmbeddingConfig::new(self.n_embed, self.d_model).init(device),
            linear: Linear2DeepConfig::new(self.d_model)
                .with_bias(true)
                .with_d_embed(self.d_embed)
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct ProductionDecisionFixed<B: Backend> {
    /// Embeddings for each rule that we want to expand.
    rule_embeddings: Embedding<B>,
    /// The linear decision function.
    linear: Linear2Deep<B>,
}

impl<B: Backend> ProductionDecisionFixed<B> {
    ///
    pub fn forward(
        &self,
        embedding: Tensor<B, 2, Float>,
        rules: Tensor<B, 2, Int>,
        activation: Activation,
    ) -> Tensor<B, 2, Float> {
        let out = self.linear.forward(embedding, activation);
        let rules = self.rule_embeddings.forward(rules);
        let rule_count = rules.dims()[1];

        println!("rules: {rules}");

        println!("out: {out}");

        let out = out.unsqueeze_dim(1);
        let out = out.repeat_dim(0, rule_count);

        // let out = rules.mul(out);

        out
    }
}

#[test]
fn test_forward() {
    use burn::backend::NdArray;

    let dev = Default::default();
    let model = ProductionDecisionFixedConfig::new(10, 5, 10).init::<NdArray>(&dev);

    let out = model.forward(
        Tensor::<NdArray, 2, Float>::from_data(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [4.0, 5.0, 4.5, 4.6, 4.7]],
            &dev,
        ),
        Tensor::<NdArray, 2, Int>::from_data([[0, 1, 2], [3, 4, 5]], &dev),
        Activation::ReLU,
    );
    println!("out: {out}");
}
