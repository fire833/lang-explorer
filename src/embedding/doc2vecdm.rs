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
    vec,
};

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    optim::{adaptor::OptimizerAdaptor, AdamW, AdamWConfig, Optimizer},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Device, Float, Int, Tensor},
    train::{TrainOutput, TrainStep},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    embedding::{GeneralEmbeddingTrainingParams, LanguageEmbedder},
    errors::LangExplorerError,
    grammar::{grammar::Grammar, NonTerminal, Terminal},
    languages::Feature,
    tooling::modules::{
        embed::pvdm::{Doc2VecDM, Doc2VecDMConfig},
        loss::nsampling::{NegativeSampling, NegativeSamplingConfig},
    },
};

pub struct Doc2VecEmbedderDM<T: Terminal, I: NonTerminal, B: AutodiffBackend> {
    // Some bullcrap.
    d1: PhantomData<T>,
    d2: PhantomData<I>,

    /// The model itself.
    model: Doc2VecDM<B>,
    loss: NegativeSampling<B>,

    device: Device<B>,

    /// RNG stuff.
    rng: ChaCha8Rng,

    optim: OptimizerAdaptor<AdamW, Doc2VecDM<B>, B>,

    params: GeneralEmbeddingTrainingParams,
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend>
    TrainStep<ProgramBatch<B>, Tensor<B, 1, Float>> for Doc2VecEmbedderDM<T, I, B>
{
    fn step(&self, item: ProgramBatch<B>) -> TrainOutput<Tensor<B, 1, Float>> {
        let logits = self
            .model
            .forward(item.documents, item.context_words, &self.params.agg);
        let loss = self.loss.forward(
            item.true_words.unsqueeze_dim(1),
            item.negative_words,
            logits,
        );

        TrainOutput::new(&self.model, loss.backward(), loss)
    }
}

#[derive(Config)]
pub struct Doc2VecDMEmbedderParams {
    /// Configuration for Adam.
    pub ada_config: AdamWConfig,
    /// The number of words within the model.
    pub n_words: usize,
    /// The number of documents within the model.
    pub n_docs: usize,
    /// The number of negative samples to update on each training run.
    #[config(default = 32)]
    pub n_neg_samples: usize,
    /// General parameters shared among all embedder params.
    pub gen_params: GeneralEmbeddingTrainingParams,
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> LanguageEmbedder<T, I, B>
    for Doc2VecEmbedderDM<T, I, B>
{
    type Document = String;
    type Word = Feature;
    type Params = Doc2VecDMEmbedderParams;

    fn new(_grammar: &Grammar<T, I>, params: Self::Params, device: Device<B>) -> Self {
        // let _uuid = grammar.generate_uuid();
        B::seed(params.gen_params.get_seed());

        // TODO: for now, just load a new model every time.
        // Custom model storage will be added soon.
        let model =
            Doc2VecDMConfig::new(params.n_words + 2, params.n_docs, params.gen_params.d_model)
                .init(&device);

        let loss = NegativeSamplingConfig::new().init(&device);

        Self {
            d1: PhantomData,
            d2: PhantomData,
            model: model,
            device,
            loss,
            optim: params.ada_config.init(),
            rng: ChaCha8Rng::seed_from_u64(params.gen_params.get_seed()),
            params: params.gen_params,
        }
    }

    fn fit(
        mut self,
        documents: &[(Self::Document, Vec<Self::Word>)],
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

        let batcher = ProgramBatcher::new();

        let mut lr: f64 = self.params.get_learning_rate();

        for num in 0..self.params.get_num_epochs() {
            println!(
                "Running epoch {} of {}",
                num + 1,
                self.params.get_num_epochs()
            );
            let mut items = vec![];

            // Adaptive learning rate
            lr *= self.params.gen_params.learning_rate_drop;
            if lr < 0.000001 {
                lr = 0.000001;
            }

            let mut counter = 0;
            for (docidx, (_, words)) in documents.iter().enumerate() {
                for (wordidx, _) in words.iter().enumerate() {
                    counter += 1;
                    let ctx_indices = get_context_indices(
                        &wordset,
                        words,
                        self.params.window_left,
                        self.params.window_right,
                        wordidx,
                        wordset.len(),
                    );

                    let negative_indices = get_negative_indices(
                        self.params.num_neg_samples,
                        &mut self.rng,
                        wordset.len() + 2,
                        wordidx,
                    );

                    let train_item =
                        ProgramTrainingItem::new(docidx, wordidx, ctx_indices, negative_indices);
                    items.push(train_item);

                    if items.len() >= self.params.get_batch_size() {
                        let moved = mem::take(&mut items);
                        let batch: ProgramBatch<B> = batcher.batch(moved, &self.device);
                        self = self.train_batch(batch, counter, lr);
                    }
                }
            }

            // Extra items need to be trained on too.
            if !items.is_empty() {
                let batch: ProgramBatch<B> = batcher.batch(items, &self.device);
                self = self.train_batch(batch, counter, lr);
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

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> Doc2VecEmbedderDM<T, I, B> {
    fn train_batch(mut self, batch: ProgramBatch<B>, counter: usize, learning_rate: f64) -> Self {
        let start = SystemTime::now();

        let train = self.step(batch);
        let grads_count = train.grads.len();
        self.model = self.optim.step(learning_rate, self.model, train.grads);

        if counter % self.params.get_display_frequency() == 0 {
            let elapsed = start.elapsed().unwrap();
            // let emb = self.model.get_embeddings().unwrap();
            let loss_data = train
                .item
                .to_data()
                .convert::<f64>()
                .to_vec()
                .unwrap_or(vec![0.0]);

            let avg = Iterator::sum::<f64>(loss_data.iter()) / loss_data.len() as f64;

            println!(
                "Training loss ({} gradients) = {} (took {} microseconds)",
                grads_count,
                avg,
                elapsed.as_micros(),
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
    context_word_indices: Vec<usize>,
    center_word_idx: usize,
    negative_sample_indices: Vec<usize>,
}

impl ProgramTrainingItem {
    fn new(
        document_idx: usize,
        center_word_idx: usize,
        context_word_indices: Vec<usize>,
        negative_sample_indices: Vec<usize>,
    ) -> Self {
        Self {
            document_idx,
            context_word_indices,
            center_word_idx,
            negative_sample_indices,
        }
    }
}

/// A batch of document/word context items that are already in their correct tensor form.
struct ProgramBatch<B: Backend> {
    documents: Tensor<B, 1, Int>,
    context_words: Tensor<B, 2, Int>,
    negative_words: Tensor<B, 2, Int>,
    true_words: Tensor<B, 1, Int>,
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
        let mut center_word_idx = vec![];
        let mut context_word_idx: Vec<Tensor<B, 1, Int>> = vec![];
        let mut negative_indices: Vec<Tensor<B, 1, Int>> = vec![];

        for item in items.iter() {
            doc_idx.push(item.document_idx as i32);
            center_word_idx.push(item.center_word_idx as i32);
            let ctx_word_indices: Vec<i32> = item
                .context_word_indices
                .iter()
                .map(|v| *v as i32)
                .collect();

            context_word_idx.push(Tensor::from_data(ctx_word_indices.as_slice(), device));
            negative_indices.push(Tensor::from_data(
                item.negative_sample_indices.as_slice(),
                device,
            ));
        }

        ProgramBatch {
            documents: Tensor::from_data(doc_idx.as_slice(), device),
            context_words: Tensor::stack::<2>(context_word_idx, 0),
            negative_words: Tensor::stack::<2>(negative_indices, 0),
            true_words: Tensor::from_data(center_word_idx.as_slice(), device),
        }
    }
}

fn get_context_indices<W: Ord>(
    wordset: &BTreeMap<W, u32>,
    document: &[W],
    window_left: usize,
    window_right: usize,
    center_word: usize,
    total_words: usize,
) -> Vec<usize> {
    let mut indices = vec![0; window_left + window_right];
    let mut counter = 0;

    for prefix in (center_word as isize - window_left as isize)..(center_word as isize) {
        if prefix < 0 {
            indices[counter] = total_words + 1;
        } else if let Some(idx) = wordset.get(document.get(prefix as usize).unwrap()) {
            indices[counter] = *idx as usize;
        } else {
            indices[counter] = total_words;
        }

        counter += 1;
    }

    for suffix in center_word + 1..(center_word + window_right + 1) {
        if suffix >= document.len() {
            indices[counter] = total_words + 1;
        } else if let Some(idx) = wordset.get(document.get(suffix).unwrap()) {
            indices[counter] = *idx as usize;
        } else {
            indices[counter] = total_words;
        }

        counter += 1;
    }

    indices
}

fn get_negative_indices<R: Rng>(
    num_negative_samples: usize,
    rng: &mut R,
    total_words: usize,
    center_word_idx: usize,
) -> Vec<usize> {
    let mut negative_samples = HashSet::new();
    while negative_samples.len() < num_negative_samples {
        let idx = rng.random::<u64>() as usize % total_words;
        if !negative_samples.contains(&idx) && idx != center_word_idx {
            negative_samples.insert(idx);
        }
    }

    negative_samples.into_iter().collect()
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
