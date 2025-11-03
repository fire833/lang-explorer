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

use burn::tensor::{backend::AutodiffBackend, Device, Tensor};

use crate::{
    embedding::LanguageEmbedder,
    errors::LangExplorerError,
    grammar::{grammar::Grammar, program::ProgramInstance, NonTerminal, Terminal},
};

pub struct GraphMAEEmbedder {}

pub struct GraphMAEEmbedderParams {}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> LanguageEmbedder<T, I, B>
    for GraphMAEEmbedder
{
    type Document = ProgramInstance<T, I>;
    type Params = GraphMAEEmbedderParams;

    fn new(_grammar: &Grammar<T, I>, _params: Self::Params, _device: Device<B>) -> Self {
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
