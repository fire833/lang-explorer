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
    errors::LangExplorerError,
    grammar::{grammar::Grammar, NonTerminal, Terminal},
};

pub mod doc2vec;

/// Main trait for creating embeddings of programs.
pub trait LanguageEmbedder<T, I, B>
where
    T: Terminal,
    I: NonTerminal,
    B: Backend,
{
    type Document;
    type Word: PartialEq + PartialOrd;
    type Params;

    /// Initializes an Embedder system. This typically involves either
    /// initializing a new model, or retrieving an already trained model
    /// instance from disk.
    fn init(grammar: &Grammar<T, I>, params: Self::Params) -> Self;

    /// Trains the embedder on the provided corpus.
    fn fit(
        &mut self,
        documents: &Vec<(Self::Document, Vec<Self::Word>)>,
    ) -> Result<(), LangExplorerError>;

    /// Creates an embedding given the current model on a new
    /// document and it's corresponding words.
    fn embed(
        &mut self,
        document: (Self::Document, Vec<Self::Word>),
    ) -> Result<Tensor<B, 1>, LangExplorerError>;
}
