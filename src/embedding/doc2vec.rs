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

use std::{collections::BTreeMap, vec};

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Device, Float, Tensor},
    train::{TrainOutput, TrainStep, ValidStep},
};

use crate::{
    embedding::{LanguageEmbedder, ProgramBatch, ProgramBatcher, TrainingItem},
    errors::LangExplorerError,
    grammar::{grammar::Grammar, NonTerminal, Terminal},
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

pub struct Doc2VecEmbedder<B: AutodiffBackend> {
    /// The model itself.
    model: Doc2VecDM<B>,
    loss: NegativeSampling<B>,

    device: Device<B>,

    optim: OptimizerAdaptor<Adam, Doc2VecDM<B>, B>,

    agg: AggregationMethod,
    window_left: usize,
    window_right: usize,
    batch_size: usize,
    n_epochs: usize,
    n_neg_samples: usize,
    learning_rate: f64,
}

impl<B: AutodiffBackend> TrainStep<ProgramBatch<B>, Tensor<B, 1, Float>> for Doc2VecEmbedder<B> {
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

impl<B: AutodiffBackend, VI, VO> ValidStep<VI, VO> for Doc2VecEmbedder<B> {
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
    /// The aggregation methdo to use.
    pub agg: AggregationMethod,
    /// The loss function to use.
    pub loss: EmbeddingLossFunction,
    /// The size of the batches fed through the model.
    pub batch_size: usize,
    /// number of epochs to train on.
    pub n_epochs: usize,
    /// the learning rate
    pub learning_rate: f64,
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> LanguageEmbedder<T, I, B>
    for Doc2VecEmbedder<B>
{
    type Document = String;
    type Word = Feature;
    type Params = Doc2VecEmbedderParams;

    fn new(_grammar: &Grammar<T, I>, params: Self::Params, device: Device<B>) -> Self {
        // let _uuid = grammar.generate_uuid();

        // TODO: for now, just load a new model every time.
        // Custom model storage will be added soon.
        let model =
            Doc2VecDMConfig::new(params.n_words + 2, params.n_docs, params.d_model).init(&device);

        let loss = NegativeSamplingConfig::new().init(&device);

        Self {
            model: model,
            device,
            loss: loss,
            optim: params.adam_config.init(),
            n_epochs: params.n_epochs,
            n_neg_samples: params.n_neg_samples,
            batch_size: params.batch_size,
            window_left: params.window_left,
            window_right: params.window_right,
            agg: params.agg,
            learning_rate: params.learning_rate,
        }
    }

    fn fit(
        mut self,
        documents: Vec<(Self::Document, Vec<Self::Word>)>,
    ) -> Result<Self, LangExplorerError> {
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

        let batcher = ProgramBatcher::new(self.n_neg_samples, wordset.len());
        let agg = self.agg.clone();
        for _ in 0..self.n_epochs {
            let mut items = vec![];

            for (docidx, (_, words)) in documents.iter().enumerate() {
                for (wordidx, _) in words.iter().enumerate() {
                    let ctx_indices = get_context_indices(
                        &wordset,
                        &words,
                        self.window_left,
                        self.window_right,
                        wordidx,
                        wordset.len(),
                    );
                    let train_item = TrainingItem::new(docidx, wordidx, ctx_indices);
                    items.push(train_item);

                    if items.len() >= self.batch_size {
                        let moved = items.drain(..).collect();
                        let batch: ProgramBatch<B> = batcher.batch(moved, &self.device);
                        self = self.train_batch(batch, agg.clone());
                    }
                }
            }
        }

        Ok(self)
    }

    fn embed(
        &mut self,
        _document: (Self::Document, Vec<Self::Word>),
    ) -> Result<Tensor<B, 1>, LangExplorerError> {
        todo!()
    }

    fn get_embeddings(&self) -> Result<Vec<Vec<f64>>, LangExplorerError> {
        todo!()
    }
}

impl<B: AutodiffBackend> Doc2VecEmbedder<B> {
    fn train_batch(mut self, batch: ProgramBatch<B>, agg: AggregationMethod) -> Self {
        let logits = self
            .model
            .forward(batch.documents, batch.context_words, &agg);
        let loss = self
            .loss
            .forward(batch.true_words, batch.negative_words, logits);

        let grads = GradientsParams::from_grads(loss.backward(), &self.model);
        self.model = self.optim.step(self.learning_rate, self.model, grads);

        self
    }
}

fn get_context_indices<W: Ord>(
    wordset: &BTreeMap<W, u32>,
    document: &Vec<W>,
    window_left: usize,
    window_right: usize,
    center_word: usize,
    total_words: usize,
) -> Vec<usize> {
    let mut indices = vec![];

    for prefix in (center_word as isize - window_left as isize)..(center_word as isize) {
        if prefix < 0 {
            indices.push(total_words + 1);
        } else {
            if let Some(idx) = wordset.get(document.get(prefix as usize).unwrap()) {
                indices.push(*idx as usize);
            } else {
                indices.push(total_words);
            }
        }
    }

    for suffix in center_word + 1..(center_word + window_right + 1) {
        if suffix >= document.len() {
            indices.push(total_words + 1);
        } else {
            if let Some(idx) = wordset.get(document.get(suffix).unwrap()) {
                indices.push(*idx as usize);
            } else {
                indices.push(total_words);
            }
        }
    }

    indices
}

#[test]
fn test_get_context_indices() {
    let mut wordset = BTreeMap::new();

    for i in 0..1001 {
        wordset.insert(i as usize, i as u32);
    }

    let doc1 = vec![4, 5, 6, 78, 89, 90, 100, 789, 521, 654, 54, 128];

    assert_eq!(
        vec![5, 6, 89, 90],
        get_context_indices(&wordset, &doc1, 2, 2, 3, wordset.len())
    );

    assert_eq!(
        vec![89, 90, 789, 521],
        get_context_indices(&wordset, &doc1, 2, 2, 6, wordset.len())
    );

    assert_eq!(
        vec![4, 5, 6, 89, 90, 100],
        get_context_indices(&wordset, &doc1, 3, 3, 3, wordset.len())
    );

    assert_eq!(
        vec![521, 654, 54, 1002, 1002, 1002],
        get_context_indices(&wordset, &doc1, 3, 3, 11, wordset.len())
    );

    assert_eq!(
        vec![1002, 5, 6, 78],
        get_context_indices(&wordset, &doc1, 1, 3, 0, wordset.len())
    );

    assert_eq!(
        vec![1002, 1002, 5, 6, 78],
        get_context_indices(&wordset, &doc1, 2, 3, 0, wordset.len())
    );

    let doc2 = vec![
        564, 536, 987, 234, 111, 743, 13, 197, 10000, 100003, 548, 435,
    ];

    assert_eq!(
        vec![1001, 548, 1002, 1002],
        get_context_indices(&wordset, &doc2, 2, 2, 11, wordset.len())
    );

    assert_eq!(
        vec![234, 111, 743, 197, 1001, 1001],
        get_context_indices(&wordset, &doc2, 3, 3, 6, wordset.len())
    )
}
