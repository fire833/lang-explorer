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
    nn::{Embedding, EmbeddingConfig},
    prelude::Backend,
    record::FileRecorder,
    tensor::{Float, Int, Tensor},
};

use crate::errors::LangExplorerError;

#[derive(Debug, Config)]
pub struct Doc2VecDBOWNSConfig {
    /// The number of word vectors.
    pub n_words: usize,
    /// The number of document vectors.
    pub n_docs: usize,
    /// The size of each vector.
    pub d_model: usize,
}

impl Doc2VecDBOWNSConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Doc2VecDBOWNS<B> {
        Doc2VecDBOWNS {
            documents: EmbeddingConfig::new(self.n_docs, self.d_model).init(device),
            hidden: EmbeddingConfig::new(self.n_words, self.d_model).init(device),
            biases: EmbeddingConfig::new(self.n_words, 1).init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct Doc2VecDBOWNS<B: Backend> {
    /// Embeddings of all documents within the corpus.
    documents: Embedding<B>,
    hidden: Embedding<B>,
    biases: Embedding<B>,
}

impl<B: Backend> Doc2VecDBOWNS<B> {
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
    /// - positive_words: `[batch_size, k]`
    /// - negative_words: `[batch_size, j]`
    /// - output: `[batch_size, n_words]`
    pub fn forward(
        &self,
        doc_inputs: Tensor<B, 1, Int>,
        positive_words: Tensor<B, 2, Int>,
        negative_words: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1, Float> {
        let num_positive_words = positive_words.shape().dims[1];
        let num_negative_words = negative_words.shape().dims[1];

        let docs = self.documents.forward(doc_inputs.unsqueeze_dim::<2>(1));

        let hidden_positive = self.hidden.forward(positive_words.clone());
        let hidden_negative = -self.hidden.forward(negative_words.clone());
        let bias_positive = self.biases.forward(positive_words);
        let bias_negative = self.hidden.forward(negative_words);

        let positives = docs.clone().repeat_dim(1, num_positive_words);
        let positives = positives.mul(hidden_positive);
        let positives = positives.sum_dim(2);
        let positives = positives.add(bias_positive);

        let positive_loss = positives.sum();

        let negatives = docs.repeat_dim(1, num_negative_words);
        let negatives = negatives.mul(hidden_negative);
        let negatives = negatives.sum_dim(2);
        let negatives = negatives.add(bias_negative);

        let negative_loss = negatives.sum();

        -positive_loss - negative_loss
    }

    /// Returns a vector of the embedding tensor. It should be structured in
    /// such a way that each n_dim elements correspond to a single embedding.
    pub fn get_embeddings(&self) -> Result<Vec<f32>, LangExplorerError> {
        let vec: Vec<f32> = self.documents.weight.to_data().convert::<f32>().to_vec()?;
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
    use burn::backend::NdArray;

    let dev = Default::default();
    let model = Doc2VecDBOWNSConfig::new(10, 5, 3).init::<NdArray>(&dev);

    let out = model.forward(
        Tensor::from_data([0, 1, 2], &dev),
        Tensor::from_data([[6, 7, 8, 9], [3, 4, 5, 6], [4, 5, 6, 7]], &dev),
        Tensor::from_data([[1], [2], [5]], &dev),
    );
    println!("{out}");
}
