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

use burn::{prelude::Backend, tensor::Tensor};

use crate::{
    embedding::LanguageEmbedder,
    errors::LangExplorerError,
    grammar::{grammar::Grammar, program::ProgramInstance, NonTerminal, Terminal},
};

pub struct Doc2VecLanguageEmbedder {}

pub struct Doc2VecLanguageEmbedderParams {}

impl<T, I, B, const D: usize> LanguageEmbedder<T, I, B, D> for Doc2VecLanguageEmbedder
where
    T: Terminal,
    I: NonTerminal,
    B: Backend,
{
    type Document = ProgramInstance<T, I>;
    type Word = u64;
    type Params = Doc2VecLanguageEmbedderParams;

    fn init(grammar: &Grammar<T, I>, params: Self::Params) -> Self {
        todo!()
    }

    fn fit(
        &mut self,
        documents: &Vec<(Self::Document, Vec<Self::Word>)>,
    ) -> Result<(), LangExplorerError> {
        todo!()
    }

    fn embed(
        &mut self,
        document: (Self::Document, Vec<Self::Word>),
    ) -> Result<Tensor<B, D>, LangExplorerError> {
        todo!()
    }
}
