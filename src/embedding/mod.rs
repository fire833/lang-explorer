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
    data::dataloader::{batcher::Batcher, DataLoader},
    prelude::Backend,
    tensor::{Int, Tensor},
};

use crate::{
    errors::LangExplorerError,
    grammar::{grammar::Grammar, NonTerminal, Terminal},
    languages::ProgramResult,
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

pub struct ProgramBatch<B: Backend> {
    documents: Tensor<B, 1, Int>,
    context_words: Tensor<B, 2, Int>,
    negative_words: Tensor<B, 2, Int>,
    true_words: Tensor<B, 1, Int>,
}

pub struct ProgramBatcher;

impl<B: Backend, D, W> Batcher<B, (D, Vec<W>, W), ProgramBatch<B>> for ProgramBatcher {
    fn batch(&self, items: Vec<(D, Vec<W>, W)>, device: &B::Device) -> ProgramBatch<B> {
        todo!()
    }
}

pub struct ProgramLoader<D, W> {
    items: Vec<(D, Vec<W>, W)>,
}

impl<B: Backend, D: Send, W: Send> DataLoader<B, ProgramBatch<B>> for ProgramLoader<D, W> {
    fn iter<'a>(
        &'a self,
    ) -> Box<dyn burn::data::dataloader::DataLoaderIterator<ProgramBatch<B>> + 'a> {
        todo!()
    }

    fn num_items(&self) -> usize {
        todo!()
    }

    fn to_device(&self, device: &B::Device) -> std::sync::Arc<dyn DataLoader<B, ProgramBatch<B>>> {
        todo!()
    }

    fn slice(
        &self,
        start: usize,
        end: usize,
    ) -> std::sync::Arc<dyn DataLoader<B, ProgramBatch<B>>> {
        todo!()
    }
}
