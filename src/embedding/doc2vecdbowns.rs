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
    fs, mem,
    time::SystemTime,
};

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    optim::{adaptor::OptimizerAdaptor, AdamW, AdamWConfig, Optimizer},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Device, Distribution, Float, Int, Tensor},
    train::{TrainOutput, TrainStep},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{
    embedding::{GeneralEmbeddingTrainingParams, LanguageEmbedder},
    errors::LangExplorerError,
    grammar::{
        grammar::Grammar,
        program::{ProgramInstance, WLKernelHashingOrder},
    },
    languages::Feature,
    tooling::modules::embed::pvdbow::{Doc2VecDBOWNS, Doc2VecDBOWNSConfig},
};

pub struct Doc2VecEmbedderDBOWNS<B: AutodiffBackend> {
    model: Doc2VecDBOWNS<B>,

    device: Device<B>,

    // RNG stuff.
    rng: ChaCha8Rng,

    optim: OptimizerAdaptor<AdamW, Doc2VecDBOWNS<B>, B>,

    params: Doc2VecDBOWNSEmbedderParams,

    display_count: usize,

    old_embeddings: Vec<f32>,

    word2idx: BTreeMap<Feature, u32>,
    idx2word: BTreeMap<u32, Feature>,
}

impl<B: AutodiffBackend> TrainStep<ProgramBatch<B>, Tensor<B, 1, Float>>
    for Doc2VecEmbedderDBOWNS<B>
{
    fn step(&self, item: ProgramBatch<B>) -> TrainOutput<Tensor<B, 1, Float>> {
        let loss = self
            .model
            .forward(item.documents, item.true_words, item.negative_words);
        TrainOutput::new(&self.model, loss.backward(), loss)
    }
}

#[derive(Config)]
pub struct Doc2VecDBOWNSEmbedderParams {
    /// Configuration for Adam.
    pub ada_config: AdamWConfig,
    /// The number of words within the model.
    pub n_words: usize,
    /// The number of documents within the model.
    pub n_docs: usize,
    /// General parameters shared among all embedder params.
    pub gen_params: GeneralEmbeddingTrainingParams,
    /// Directory where the model directory is.
    pub model_dir: String,
}

impl<B: AutodiffBackend> LanguageEmbedder<B> for Doc2VecEmbedderDBOWNS<B> {
    type Document = ProgramInstance;
    type Params = Doc2VecDBOWNSEmbedderParams;

    fn new(grammar: &Grammar, params: Self::Params, device: Device<B>) -> Self {
        B::seed(params.gen_params.get_seed());

        let model_location = format!(
            "{}/embeddings/{}",
            params.model_dir,
            grammar.generate_location()
        );

        let mbase =
            Doc2VecDBOWNSConfig::new(params.n_words + 2, params.n_docs, params.gen_params.d_model)
                .init(&device);

        let model = match (
            fs::metadata(&model_location),
            params.gen_params.get_create_new_model(),
        ) {
            (Ok(_), true) => mbase,
            (Ok(m), false) if m.is_file() => match Doc2VecDBOWNS::load_file(
                mbase,
                model_location,
                &params.gen_params.get_model_recorder(),
                &device,
            ) {
                Ok(m) => m,
                Err(e) => panic!("{}", e),
            },
            (Ok(_), false) => {
                println!("model is not a regular file, reverting to starting from scratch");
                mbase
            }
            (Err(_), _) => mbase,
        };

        Self {
            model,
            device,
            optim: params.ada_config.init(),
            rng: ChaCha8Rng::seed_from_u64(params.gen_params.get_seed()),
            params,
            old_embeddings: vec![],
            display_count: 0,
            word2idx: BTreeMap::new(),
            idx2word: BTreeMap::new(),
        }
    }

    fn fit(mut self, documents: &[Self::Document]) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        let mut counter: u32 = 0;
        documents.iter().for_each(|doc| {
            let words = &doc.extract_words_wl_kernel(
                5,
                WLKernelHashingOrder::ParentSelfChildrenOrdered,
                false,
                false,
            );

            words.iter().for_each(|word| {
                if !self.word2idx.contains_key(word) {
                    self.word2idx.insert(*word, counter);
                    self.idx2word.insert(counter, *word);
                    counter += 1;
                }
            })
        });

        let batcher = ProgramBatcher::new();
        let mut lr: f64 = self.params.gen_params.get_learning_rate();
        let mut counter = 0;

        for num in 0..self.params.gen_params.get_num_epochs() {
            println!(
                "Running epoch {} of {}",
                num + 1,
                self.params.gen_params.get_num_epochs()
            );
            let mut items = vec![];

            // Adaptive learning rate
            lr *= self.params.gen_params.gen_params.learning_rate_drop;
            if lr < self.params.gen_params.gen_params.min_learning_rate {
                lr = self.params.gen_params.gen_params.min_learning_rate;
            }

            for (docidx, doc) in documents.iter().enumerate() {
                let words = &doc.extract_words_wl_kernel(
                    5,
                    WLKernelHashingOrder::ParentSelfChildrenOrdered,
                    false,
                    false,
                );

                let set = words.iter().cloned().collect::<HashSet<Feature>>();
                for _ in words.iter().enumerate() {
                    counter += 1;
                    let positive_indices =
                        super::get_positive_indices(&self.word2idx, words, 1, &mut self.rng);

                    let negative_indices = super::get_negative_indices(
                        &self.idx2word,
                        &set,
                        self.params.gen_params.num_neg_samples,
                        &mut self.rng,
                    );

                    let train_item =
                        ProgramTrainingItem::new(docidx, positive_indices, negative_indices);
                    items.push(train_item);

                    if items.len() >= self.params.gen_params.get_batch_size() {
                        let moved = mem::take(&mut items);
                        let batch: ProgramBatch<B> = batcher.batch(moved, &self.device);
                        self.train_batch(batch, counter, lr);
                    }
                }
            }

            // Extra items need to be trained on too.
            if !items.is_empty() {
                let batch: ProgramBatch<B> = batcher.batch(items, &self.device);
                self.train_batch(batch, counter, lr);
            }
        }

        if self.params.gen_params.get_save_model() {
            let recorder = self.params.gen_params.get_model_recorder();
            self.model
                .clone()
                .save_file(&self.params.model_dir, &recorder)?;
        }

        Ok(self)
    }

    fn embed(
        &mut self,
        document: Self::Document,
    ) -> Result<Tensor<B, 1, Float>, LangExplorerError> {
        let vec: Tensor<B, 1, Float> = Tensor::random(
            [self.params.gen_params.d_model],
            Distribution::Uniform(0.0, 1.0),
            &self.device,
        );

        let batcher = ProgramBatcher::new();
        let mut lr: f64 = self.params.gen_params.get_learning_rate();
        let mut counter = 0;
        let words = &document.extract_words_wl_kernel(
            5,
            WLKernelHashingOrder::ParentSelfChildrenOrdered,
            false,
            false,
        );
        let set = words.iter().cloned().collect::<HashSet<Feature>>();

        for _num in 0..5 {
            let mut items = vec![];

            // Adaptive learning rate
            lr *= self.params.gen_params.gen_params.learning_rate_drop;
            if lr < 0.000001 {
                lr = 0.000001;
            }

            for _ in words.iter().enumerate() {
                counter += 1;
                let positive_indices =
                    super::get_positive_indices(&self.word2idx, words, 1, &mut self.rng);

                let negative_indices = super::get_negative_indices(
                    &self.idx2word,
                    &set,
                    self.params.gen_params.num_neg_samples,
                    &mut self.rng,
                );

                let train_item = ProgramTrainingItem::new(0, positive_indices, negative_indices);
                items.push(train_item);

                if items.len() >= self.params.gen_params.get_batch_size() {
                    let moved = mem::take(&mut items);
                    let batch: ProgramBatch<B> = batcher.batch(moved, &self.device);
                    self.train_batch(batch, counter, lr);
                }
            }

            // Extra items need to be trained on too.
            if !items.is_empty() {
                let batch: ProgramBatch<B> = batcher.batch(items, &self.device);
                self.train_batch(batch, counter, lr);
            }
        }

        Ok(vec)
    }

    fn get_embeddings(&self) -> Result<Vec<f32>, LangExplorerError> {
        self.model.get_embeddings()
    }
}

impl<B: AutodiffBackend> Doc2VecEmbedderDBOWNS<B> {
    fn train_batch(&mut self, batch: ProgramBatch<B>, counter: usize, learning_rate: f64) {
        let start = SystemTime::now();

        let train = self.step(batch);
        let grads_count = train.grads.len();
        self.model = self
            .optim
            .step(learning_rate, self.model.clone(), train.grads);

        if counter % self.params.gen_params.get_display_frequency() == 0 {
            let elapsed = start.elapsed().unwrap();
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

            self.display_count += 1;

            let _ = super::save_embeddings_as_csv(
                &emb,
                self.params.gen_params.d_model,
                format!(
                    "lang-explorer-python/results/temporal/vectors_{:04}.csv",
                    self.display_count
                ),
            );

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

/// A batch of document/word context items that are already
/// in their correct tensor form.
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
