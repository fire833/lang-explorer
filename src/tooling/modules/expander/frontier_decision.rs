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
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Embedding, EmbeddingConfig,
    },
    prelude::Backend,
    tensor::{Float, Tensor},
};

use crate::grammar::{elem::GrammarElement, NonTerminal, Terminal};

#[derive(Debug, Config)]
pub struct FrontierDecisionConfig {}

impl FrontierDecisionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FrontierDecision<B> {
        FrontierDecision {
            symbols_embeddings: EmbeddingConfig::new(0, 0).init(device),
            decision: MultiHeadAttentionConfig::new(0, 5).init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct FrontierDecision<B: Backend> {
    /// Embeddings for all symbols in the grammar.
    /// More specifically, this is |V ∪ T ∪ λ|.
    symbols_embeddings: Embedding<B>,
    /// The decision attention head.
    decision: MultiHeadAttention<B>,
}

impl<B: Backend> FrontierDecision<B> {
    pub fn forward<T: Terminal, I: NonTerminal>(
        &self,
        frontier: Vec<GrammarElement<T, I>>,
    ) -> Tensor<B, 2, Float> {
        // self.decision.forward(MhaInput::new(query, key, value))
        todo!()
    }
}

#[test]
fn test_forward() {}
