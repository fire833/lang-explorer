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

use std::collections::HashSet;

use burn::{
    prelude::Backend,
    record::{BinGzFileRecorder, HalfPrecisionSettings},
    tensor::Tensor,
};

use crate::{
    embedding::LanguageEmbedder,
    errors::LangExplorerError,
    grammar::{grammar::Grammar, program::ProgramInstance, NonTerminal, Terminal},
    tooling::modules::embed::pvdm::{Doc2VecDM, Doc2VecDMConfig},
};

pub struct Doc2VecLanguageEmbedder<B: Backend> {
    /// The model itself.
    model: Doc2VecDM<B>,

    /// Recorder to read/write models to disk.
    recorder: BinGzFileRecorder<HalfPrecisionSettings>,
}

pub struct Doc2VecLanguageEmbedderParams {
    /// The number of words within the model.
    pub n_words: usize,
    /// The number of documents within the model.
    pub n_docs: usize,
    /// The dimension of embeddings within the model.
    pub d_model: usize,
    /// The window size when choosing context words
    /// to predict the center word.
    pub window: usize,
    /// number of epochs to train on.
    pub n_epochs: usize,
}

impl<T, I, B, const D: usize> LanguageEmbedder<T, I, B, D> for Doc2VecLanguageEmbedder<B>
where
    T: Terminal,
    I: NonTerminal,
    B: Backend,
{
    type Document = ProgramInstance<T, I>;
    type Word = u64;
    type Params = Doc2VecLanguageEmbedderParams;

    fn init(grammar: &Grammar<T, I>, params: Self::Params) -> Self {
        let _uuid = grammar.generate_uuid();
        let device = Default::default();

        // TODO: for now, just load a new model every time.
        // Custom model storage will be added soon.
        let model =
            Doc2VecDMConfig::new(params.n_words, params.n_docs, params.d_model).init(&device);

        Self {
            model: model,
            recorder: BinGzFileRecorder::new(),
        }
    }

    fn fit(
        &mut self,
        documents: &Vec<(Self::Document, Vec<Self::Word>)>,
    ) -> Result<(), LangExplorerError> {
        let mut wordset: HashSet<u64> = HashSet::new();

        documents.iter().for_each(|doc| {
            doc.1.iter().for_each(|word| {
                wordset.insert(*word);
            })
        });

        // this will be useful later
        // let negative_indices = Tensor::<B, 2, Int>::random(
        //     [batch_size, k],
        //     Distribution::Uniform(0.0, n_words as f64),
        //     device,
        // );

        todo!()
    }

    fn embed(
        &mut self,
        _document: (Self::Document, Vec<Self::Word>),
    ) -> Result<Tensor<B, D>, LangExplorerError> {
        todo!()
    }
}
