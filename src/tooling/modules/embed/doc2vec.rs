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
    config::Config,
    module::Module,
    prelude::Backend,
    tensor::backend::AutodiffBackend,
    train::{TrainStep, ValidStep},
};

use crate::tooling::modules::{
    embed::{
        pvdbow::{Doc2VecDBOW, Doc2VecDBOWConfig},
        pvdm::{Doc2VecDM, Doc2VecDMConfig},
    },
    loss::nsampling::{NegativeSampling, NegativeSamplingConfig},
};

#[derive(Debug, Config)]
pub enum Doc2VecType {
    DBOW,
    DM,
}

#[derive(Debug, Config)]
pub struct Doc2VecConfig {
    /// The seed for things.
    #[config(default = 1000)]
    pub seed: u64,
    /// The type of model to instantiate.
    pub model_type: Doc2VecType,
    /// The number of word vectors.
    pub n_words: usize,
    /// The number of document vectors.
    pub n_docs: usize,
    /// The size of each vector.
    pub d_model: usize,
}

impl Doc2VecConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Doc2Vec<B> {
        let ns = NegativeSamplingConfig::new().init(device);

        match self.model_type {
            Doc2VecType::DBOW => Doc2Vec {
                pvbdow: Some(
                    Doc2VecDBOWConfig::new(self.n_words, self.n_docs, self.d_model).init(device),
                ),
                pvdm: None,
                sample: ns,
            },
            Doc2VecType::DM => Doc2Vec {
                pvbdow: None,
                pvdm: Some(
                    Doc2VecDMConfig::new(self.n_words, self.n_docs, self.d_model).init(device),
                ),
                sample: ns,
            },
        }
    }
}

impl<B: AutodiffBackend, TI, TO> TrainStep<TI, TO> for Doc2Vec<B> {
    fn step(&self, item: TI) -> burn::train::TrainOutput<TO> {
        todo!()
    }
}

impl<B: Backend, VI, VO> ValidStep<VI, VO> for Doc2Vec<B> {
    fn step(&self, item: VI) -> VO {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct Doc2Vec<B: Backend> {
    /// Use a DBOW model.
    pvbdow: Option<Doc2VecDBOW<B>>,

    /// Use a ditributed memory model.
    pvdm: Option<Doc2VecDM<B>>,

    sample: NegativeSampling<B>,
}
