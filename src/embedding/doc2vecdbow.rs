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

use std::{
    collections::{BTreeMap, HashSet},
    marker::PhantomData,
    mem,
    time::SystemTime,
};

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    optim::{adaptor::OptimizerAdaptor, AdamW, AdamWConfig, Optimizer},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Device, Float, Int, Tensor},
    train::{TrainOutput, TrainStep},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{
    embedding::{GeneralEmbeddingTrainingParams, LanguageEmbedder},
    errors::LangExplorerError,
    grammar::{grammar::Grammar, NonTerminal, Terminal},
    languages::Feature,
    tooling::modules::{
        embed::pvdbow::{Doc2VecDBOW, Doc2VecDBOWConfig},
        loss::nsampling::{NegativeSampling, NegativeSamplingConfig},
    },
};

pub struct Doc2VecEmbedderDBOW<T: Terminal, I: NonTerminal, B: AutodiffBackend> {
    // Some bullcrap.
    d1: PhantomData<T>,
    d2: PhantomData<I>,

    model: Doc2VecDBOW<B>,
    loss: NegativeSampling<B>,

    device: Device<B>,

    // RNG stuff.
    rng: ChaCha8Rng,

    optim: OptimizerAdaptor<AdamW, Doc2VecDBOW<B>, B>,

    params: GeneralEmbeddingTrainingParams,

    old_embeddings: Vec<f32>,
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend>
    TrainStep<ProgramBatch<B>, Tensor<B, 1, Float>> for Doc2VecEmbedderDBOW<T, I, B>
{
    fn step(&self, item: ProgramBatch<B>) -> TrainOutput<Tensor<B, 1, Float>> {
        let logits = self.model.forward(item.documents);
        let loss = self
            .loss
            .forward(item.true_words, item.negative_words, logits);

        TrainOutput::new(&self.model, loss.backward(), loss)
    }
}

#[derive(Config)]
pub struct Doc2VecDBOWEmbedderParams {
    /// Configuration for Adam.
    pub ada_config: AdamWConfig,
    /// The number of words within the model.
    pub n_words: usize,
    /// The number of documents within the model.
    pub n_docs: usize,
    /// General parameters shared among all embedder params.
    pub gen_params: GeneralEmbeddingTrainingParams,
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> LanguageEmbedder<T, I, B>
    for Doc2VecEmbedderDBOW<T, I, B>
{
    type Document = String;
    type Word = Feature;
    type Params = Doc2VecDBOWEmbedderParams;

    fn new(_grammar: &Grammar<T, I>, params: Self::Params, device: Device<B>) -> Self {
        B::seed(params.gen_params.get_seed());

        let model =
            Doc2VecDBOWConfig::new(params.n_words + 2, params.n_docs, params.gen_params.d_model)
                .init(&device);

        Self {
            d1: PhantomData,
            d2: PhantomData,
            model: model,
            loss: NegativeSamplingConfig::new().init(&device),
            device: device,
            optim: params.ada_config.init(),
            rng: ChaCha8Rng::seed_from_u64(params.gen_params.get_seed()),
            params: params.gen_params,
            old_embeddings: vec![],
        }
    }

    fn fit(
        mut self,
        documents: &[(Self::Document, Vec<Self::Word>)],
    ) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        let mut wordset: BTreeMap<Self::Word, u32> = BTreeMap::new();
        // I hate this
        let mut wordvec: BTreeMap<u32, Self::Word> = BTreeMap::new();

        let mut counter: u32 = 0;
        documents.iter().for_each(|doc| {
            doc.1.iter().for_each(|word| {
                if !wordset.contains_key(word) {
                    wordset.insert(*word, counter);
                    wordvec.insert(counter, *word);
                    counter += 1;
                }
            })
        });

        let batcher = ProgramBatcher::new();

        for num in 0..self.params.get_num_epochs() {
            println!(
                "Running epoch {} of {}",
                num + 1,
                self.params.get_num_epochs()
            );
            let mut items = vec![];

            // Adaptive learning rate
            self.params.gen_params.learning_rate *= self.params.gen_params.learning_rate_drop;
            if self.params.get_learning_rate() < 0.000001 {
                self.params.gen_params.learning_rate = 0.000001;
            }

            let mut counter = 0;
            for (docidx, (_, words)) in documents.iter().enumerate() {
                let set = HashSet::from_iter(words.iter().cloned());
                for (_, _) in words.iter().enumerate() {
                    counter += 1;
                    let positive_indices =
                        super::get_positive_indices(&wordset, words, 1, &mut self.rng);

                    let negative_indices = super::get_negative_indices(
                        &wordvec,
                        &set,
                        self.params.num_neg_samples,
                        &mut self.rng,
                    );

                    let train_item =
                        ProgramTrainingItem::new(docidx, positive_indices, negative_indices);
                    items.push(train_item);

                    if items.len() >= self.params.get_batch_size() {
                        let moved = mem::take(&mut items);
                        let batch: ProgramBatch<B> = batcher.batch(moved, &self.device);
                        self = self.train_batch(batch, counter);
                    }
                }
            }

            // Extra items need to be trained on too.
            if !items.is_empty() {
                let batch: ProgramBatch<B> = batcher.batch(items, &self.device);
                self = self.train_batch(batch, counter);
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

    fn get_embeddings(&self) -> Result<Vec<f32>, LangExplorerError> {
        self.model.get_embeddings()
    }
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> Doc2VecEmbedderDBOW<T, I, B> {
    fn train_batch(mut self, batch: ProgramBatch<B>, counter: usize) -> Self {
        let start = SystemTime::now();

        let train = self.step(batch);
        let grads_count = train.grads.len();
        self.model = self
            .optim
            .step(self.params.get_learning_rate(), self.model, train.grads);

        if counter % 1000 == 0 {
            let elapsed = start.elapsed().unwrap();
            // let emb = self.model.get_embeddings().unwrap();
            let loss_data = train
                .item
                .to_data()
                .convert::<f64>()
                .to_vec()
                .unwrap_or(vec![0.0]);

            let avg = Iterator::sum::<f64>(loss_data.iter()) / loss_data.len() as f64;

            let emb = self.get_embeddings().unwrap();

            let mut absdiff: f32 = 0.0;
            for (idx, val) in self.old_embeddings.iter().enumerate() {
                absdiff += f32::abs(*val - *emb.get(idx).unwrap());
            }

            self.old_embeddings = emb;

            println!(
                "Training loss ({} gradients) = {} (took {} microseconds) absolute difference: {}",
                grads_count,
                avg,
                elapsed.as_micros(),
                absdiff,
                // &emb[0..128],
            );
        }

        self
    }
}

/// A training item represents a single item in a batch that should be trained on.
#[derive(Debug, Clone)]
struct ProgramTrainingItem {
    document_idx: usize,
    true_word_indices: Vec<usize>,
    negative_sample_indices: Vec<usize>,
}

impl ProgramTrainingItem {
    fn new(
        document_idx: usize,
        true_word_indices: Vec<usize>,
        negative_sample_indices: Vec<usize>,
    ) -> Self {
        Self {
            document_idx,
            true_word_indices,
            negative_sample_indices,
        }
    }
}

/// A batch of document/word context items that are already in their correct tensor form.
struct ProgramBatch<B: Backend> {
    documents: Tensor<B, 1, Int>,
    negative_words: Tensor<B, 2, Int>,
    true_words: Tensor<B, 2, Int>,
}

#[derive(Debug, Clone)]
struct ProgramBatcher {}

impl ProgramBatcher {
    fn new() -> Self {
        Self {}
    }
}

impl<B: Backend> Batcher<B, ProgramTrainingItem, ProgramBatch<B>> for ProgramBatcher {
    fn batch(&self, items: Vec<ProgramTrainingItem>, device: &B::Device) -> ProgramBatch<B> {
        let mut doc_idx = vec![];
        let mut true_word_indices: Vec<Tensor<B, 1, Int>> = vec![];
        let mut negative_indices: Vec<Tensor<B, 1, Int>> = vec![];

        for item in items.iter() {
            doc_idx.push(item.document_idx as i32);
            true_word_indices.push(Tensor::from_data(item.true_word_indices.as_slice(), device));
            negative_indices.push(Tensor::from_data(
                item.negative_sample_indices.as_slice(),
                device,
            ));
        }

        ProgramBatch {
            documents: Tensor::from_data(doc_idx.as_slice(), device),
            negative_words: Tensor::stack::<2>(negative_indices, 0),
            true_words: Tensor::stack::<2>(true_word_indices, 0),
        }
    }
}
