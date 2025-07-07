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

// Notes: https://github.com/piskvorky/gensim/blob/develop/gensim/models/doc2vec.py
// https://github.com/cbowdon/doc2vec-pytorch/blob/master/doc2vec.py
// https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html

use std::path::PathBuf;

use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::Backend,
    record::FileRecorder,
    tensor::{Float, Int, Tensor},
};

#[allow(unused)]
use burn::backend::NdArray;

use crate::errors::LangExplorerError;

#[derive(Debug, Config)]
pub struct Doc2VecDBOWConfig {
    /// The number of word vectors.
    pub n_words: usize,
    /// The number of document vectors.
    pub n_docs: usize,
    /// The size of each vector.
    pub d_model: usize,
}

impl Doc2VecDBOWConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Doc2VecDBOW<B> {
        Doc2VecDBOW {
            documents: EmbeddingConfig::new(self.n_docs, self.d_model).init(device),
            hidden: LinearConfig::new(self.d_model, self.n_words)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct Doc2VecDBOW<B: Backend> {
    /// Embeddings of all documents within the corpus.
    documents: Embedding<B>,
    /// The hidden layer to compute output logits.
    hidden: Linear<B>,
}

impl<B: Backend> Doc2VecDBOW<B> {
    /// Applies the forward pass to the input tensor.
    ///
    /// More specifically, takes a list of document vectors indices.
    /// The corresponding vector will be passed through the classifier
    /// and for each batch item and array of logits of size `n_words` will
    /// be returned.
    ///
    /// # Shapes
    ///
    /// - doc_inputs: `[batch_size]`
    /// - output: `[batch_size, n_words]`
    pub fn forward(&self, doc_inputs: Tensor<B, 1, Int>) -> Tensor<B, 2, Float> {
        let docs = self.documents.forward(doc_inputs.unsqueeze_dim::<2>(0));
        let out = self.hidden.forward(docs);
        return out.squeeze::<2>(0);
    }

    /// Save the current embeddings to a separate file.
    pub fn save_embeddings<FR: FileRecorder<B>, PB: Into<PathBuf>>(
        &self,
        file_path: PB,
        recorder: &FR,
    ) -> Result<(), LangExplorerError> {
        match self.documents.clone().save_file(file_path, recorder) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

#[test]
fn test_forward() {
    let dev = Default::default();
    let model = Doc2VecDBOWConfig::new(10, 5, 3).init::<NdArray>(&dev);

    let out = model.forward(Tensor::from_data([0, 1, 2, 3, 4], &dev));
    println!("{out}");
}
