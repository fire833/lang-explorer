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
    tensor::{Float, Tensor},
};

#[derive(Debug, Config)]
pub struct ProductionDecisionConfig {
    /// The number of dimensions of embeddings.
    d_model: usize,

    /// The number of embeddings to store. This will typically
    /// correspond to the number of rules within the grammar, since
    /// we will want to score the best rule to expand.
    n_embed: usize,
}

impl ProductionDecisionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ProductionDecision<B> {
        ProductionDecision {
            rule_embeddings: EmbeddingConfig::new(self.n_embed, self.d_model).init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct ProductionDecision<B: Backend> {
    /// Embeddings for each rule that we want to expand.
    rule_embeddings: Embedding<B>,
}

impl<B: Backend> ProductionDecision<B> {
    pub fn forward(&self) -> Tensor<B, 2, Float> {
        todo!()
    }
}

#[test]
fn test_forward() {}
