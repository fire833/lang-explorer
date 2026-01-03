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
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Device, Tensor},
};

use crate::{
    embedding::LanguageEmbedder,
    errors::LangExplorerError,
    grammar::{grammar::Grammar, program::ProgramInstance},
    tooling::modules::embed::gnn::GraphNeuralNet,
};

pub struct GraphMAEEmbedder<B: Backend> {
    encoder: GraphNeuralNet<B>,
    decoder: GraphNeuralNet<B>,
}

pub struct GraphMAEEmbedderParams {}

impl<B: AutodiffBackend> LanguageEmbedder<B> for GraphMAEEmbedder<B> {
    type Document = ProgramInstance;
    type Params = GraphMAEEmbedderParams;

    fn new(_grammar: &Grammar, _params: Self::Params, _device: Device<B>) -> Self {
        todo!()
    }

    fn fit(self, _documents: &[Self::Document]) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        todo!()
    }

    fn embed(&mut self, _document: Self::Document) -> Result<Tensor<B, 1>, LangExplorerError> {
        todo!()
    }

    fn get_embeddings(&self) -> Result<Vec<f32>, LangExplorerError> {
        todo!()
    }
}
