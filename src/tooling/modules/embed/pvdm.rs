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
// https://github.com/cbowdon/doc2vec-pytorch/blob/master/doc2vec.ipynb

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

use crate::{errors::LangExplorerError, tooling::modules::embed::AggregationMethod};

#[derive(Debug, Config)]
pub struct Doc2VecDMConfig {
    /// The number of word vectors.
    pub n_words: usize,
    /// The number of document vectors.
    pub n_docs: usize,
    /// The size of each vector.
    pub d_model: usize,
}

impl Doc2VecDMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Doc2VecDM<B> {
        Doc2VecDM {
            words: EmbeddingConfig::new(self.n_words, self.d_model).init(device),
            documents: EmbeddingConfig::new(self.n_docs, self.d_model).init(device),
            hidden: LinearConfig::new(self.d_model, self.n_words)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct Doc2VecDM<B: Backend> {
    /// Embeddings of all words within the corpus.
    words: Embedding<B>,
    /// Embeddings of all documents within the corpus.
    documents: Embedding<B>,
    /// The hidden layer to compute output logits.
    hidden: Linear<B>,
}

impl<B: Backend> Doc2VecDM<B> {
    /// Applies the forward pass to the input tensor.
    ///
    /// More specifically, the forward pass takes a list of documents indices
    /// and a list of a list of context word indices (of size word_context_size)
    /// that will be averaged/added together and added to the document vector.
    /// The completed vector will be passed through the classifier and for each
    /// batch item and array of logits of size `n_words` will be returned.
    ///
    /// # Shapes
    ///
    /// - doc_inputs: `[batch_size]`
    /// - word_inputs: `[batch_size, word_context_size]`
    /// - output: `[batch_size, n_words]`
    pub fn forward(
        &self,
        doc_inputs: Tensor<B, 1, Int>,
        word_inputs: Tensor<B, 2, Int>,
        agg: &AggregationMethod,
    ) -> Tensor<B, 2, Float> {
        let docs = self
            .documents
            .forward(doc_inputs.unsqueeze_dim::<2>(0))
            .squeeze::<2>(0);
        let words = self.words.forward(word_inputs);

        let words_summed = match agg {
            AggregationMethod::Average => words.mean_dim(1),
            AggregationMethod::Sum => words.sum_dim(1),
        }
        .squeeze::<2>(1);

        let int = docs.add(words_summed);
        let out = self.hidden.forward(int);
        return out;
    }

    /// Returns a vector of the embedding tensor. It should be structured in
    /// such a way that each n_dim elements correspond to a single embedding.
    pub fn get_embeddings(&self) -> Result<Vec<f64>, LangExplorerError> {
        let vec: Vec<f64> = self.documents.weight.to_data().convert::<f64>().to_vec()?;
        Ok(vec)
    }

    /// Save the current embeddings to a separate file.
    #[allow(unused)]
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
    let model = Doc2VecDMConfig::new(10, 5, 3).init::<NdArray>(&dev);

    let out = model.forward(
        Tensor::from_data([0, 1, 2, 3, 4], &dev),
        Tensor::from_data(
            [
                [0, 1, 2, 3],
                [2, 3, 4, 0],
                [3, 4, 5, 6],
                [9, 8, 7, 6],
                [5, 7, 9, 0],
            ],
            &dev,
        ),
        &AggregationMethod::Sum,
    );
    println!("{out}");
}
