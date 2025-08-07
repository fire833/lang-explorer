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

use std::collections::BTreeMap;

use burn::{
    config::Config,
    module::AutodiffModule,
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams},
    tensor::{backend::AutodiffBackend, Device, Float, Int, Tensor},
    train::{TrainOutput, TrainStep, ValidStep},
};

use crate::{
    embedding::{LanguageEmbedder, ProgramBatch},
    errors::LangExplorerError,
    grammar::{grammar::Grammar, program::ProgramInstance, NonTerminal, Terminal},
    languages::Feature,
    tooling::modules::{
        embed::{
            loss::EmbeddingLossFunction,
            pvdm::{Doc2VecDM, Doc2VecDMConfig},
            AggregationMethod,
        },
        loss::nsampling::{NegativeSampling, NegativeSamplingConfig},
    },
};

pub struct Doc2VecEmbedder<M: AutodiffModule<B>, B: AutodiffBackend> {
    /// The model itself.
    model: Doc2VecDM<B>,
    loss: NegativeSampling<B>,

    device: Device<B>,

    optim: OptimizerAdaptor<Adam, M, B>,

    strategy: Doc2VecTrainingStrategy,
    agg: AggregationMethod,
    window_left: usize,
    window_right: usize,
    batch_size: usize,
    n_epochs: usize,
}

#[derive(Debug, Config)]
pub enum Doc2VecTrainingStrategy {
    AllDocsAllSubwords,
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> TrainStep<ProgramBatch<B>, Tensor<B, 1, Float>>
    for Doc2VecEmbedder<M, B>
{
    fn step(&self, item: ProgramBatch<B>) -> burn::train::TrainOutput<Tensor<B, 1, Float>> {
        let logits = self
            .model
            .forward(item.documents, item.context_words, &self.agg);
        let loss = self
            .loss
            .forward(item.true_words, item.negative_words, logits);

        TrainOutput::new(&self.model, loss.backward(), loss)
    }
}

impl<M: AutodiffModule<B>, B: AutodiffBackend, VI, VO> ValidStep<VI, VO> for Doc2VecEmbedder<M, B> {
    fn step(&self, _item: VI) -> VO {
        todo!()
    }
}

#[derive(Config)]
pub struct Doc2VecEmbedderParams {
    /// Configuration for Adam.
    pub adam_config: AdamConfig,
    /// The number of words within the model.
    pub n_words: usize,
    /// The number of documents within the model.
    pub n_docs: usize,
    /// The dimension of embeddings within the model.
    pub d_model: usize,
    /// The number of negative samples to update on each training run.
    pub n_neg_samples: usize,
    /// The number of words to the left of the center word
    /// to predict on.
    pub window_left: usize,
    /// The number of words to the right of the center word
    /// to predict on.
    pub window_right: usize,
    /// The training strategy to use.
    pub strategy: Doc2VecTrainingStrategy,
    /// The aggregation methdo to use.
    pub agg: AggregationMethod,
    /// The loss function to use.
    pub loss: EmbeddingLossFunction,
    /// The size of the batches fed through the model.
    pub batch_size: usize,
    /// number of epochs to train on.
    pub n_epochs: usize,
}

impl<T: Terminal, I: NonTerminal, M: AutodiffModule<B>, B: AutodiffBackend>
    LanguageEmbedder<T, I, B> for Doc2VecEmbedder<M, B>
{
    type Document = ProgramInstance<T, I>;
    type Word = Feature;
    type Params = Doc2VecEmbedderParams;

    fn init(_grammar: &Grammar<T, I>, params: Self::Params, device: Device<B>) -> Self {
        // let _uuid = grammar.generate_uuid();

        // TODO: for now, just load a new model every time.
        // Custom model storage will be added soon.
        let model =
            Doc2VecDMConfig::new(params.n_words, params.n_docs, params.d_model).init(&device);

        let loss = NegativeSamplingConfig::new().init(&device);

        Self {
            model: model,
            device,
            loss: loss,
            optim: params.adam_config.init(),
            n_epochs: params.n_epochs,
            batch_size: params.batch_size,
            window_left: params.window_left,
            window_right: params.window_right,
            strategy: params.strategy,
            agg: params.agg,
        }
    }

    fn fit(
        &mut self,
        documents: &Vec<(Self::Document, Vec<Self::Word>)>,
    ) -> Result<(), LangExplorerError> {
        let mut wordset: BTreeMap<Self::Word, u32> = BTreeMap::new();

        let mut counter: u32 = 0;
        documents.iter().for_each(|doc| {
            doc.1.iter().for_each(|word| {
                if !wordset.contains_key(word) {
                    wordset.insert(*word, counter);
                    counter += 1;
                }
            })
        });

        for epoch in 0..self.n_epochs {
            match self.strategy {
                Doc2VecTrainingStrategy::AllDocsAllSubwords => {
                    for (docidx, doc) in documents.iter().enumerate() {
                        for word in doc.1.iter() {
                            let true_idx = *wordset.get(word).unwrap();
                            let docs = Tensor::from_ints([[docidx as i32]], &self.device);
                            let words = self.build_word_indices(&wordset, vec![(&doc.1, true_idx)]);
                            let logits = self.model.forward(docs, words, &self.agg);
                            let loss = self.loss.forward(
                                Tensor::from_ints([[true_idx as i32]], &self.device),
                                Tensor::from_ints([[true_idx as i32]], &self.device), // TODO fix this broken crap
                                logits,
                            );

                            let grads = loss.backward();
                            let mut grads = GradientsParams::from_grads(grads, &self.model);

                            // self.model = self.optim.step(0.01, &self.model, grads);
                        }
                    }
                }
            }
        }

        todo!()
    }

    fn embed(
        &mut self,
        _document: (Self::Document, Vec<Self::Word>),
    ) -> Result<Tensor<B, 1>, LangExplorerError> {
        todo!()
    }
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> Doc2VecEmbedder<M, B> {
    fn build_word_indices<W>(
        &self,
        word_map: &BTreeMap<W, u32>,
        // The vector of document words, and the center word for each batch element.
        batch_elements: Vec<(&Vec<W>, u32)>,
    ) -> Tensor<B, 2, Int> {
        // let indices = vec![];

        Tensor::from_ints([[1], [2]], &self.device)
    }
}
