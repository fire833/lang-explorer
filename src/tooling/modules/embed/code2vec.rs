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
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Distribution, Float, Int, Tensor},
};

#[derive(Debug, Config)]
#[allow(unused)]
pub struct Code2VecConfig {
    /// The number of terminals.
    pub n_terminals: usize,
    /// The number of code paths to learn on.
    pub n_paths: usize,
    /// The number of output labels.
    pub n_labels: usize,
    /// The size of each terminal vector.
    pub d_terminal: usize,
    /// The size of each path vector.
    pub d_path: usize,
    /// The dimension of the encoded vectors after
    /// passing through the linear input.
    pub d_model: usize,
}

impl Code2VecConfig {
    #[allow(unused)]
    pub fn init<B: Backend>(&self, device: &B::Device) -> Code2Vec<B> {
        Code2Vec {
            terminals: EmbeddingConfig::new(self.n_terminals, self.d_terminal).init(device),
            paths: EmbeddingConfig::new(self.n_paths, self.d_path).init(device),
            attention: Tensor::random([1, self.d_model], Distribution::Uniform(-1.0, 1.0), device),
            hidden_in: LinearConfig::new(2 * self.d_terminal + self.d_path, self.d_model)
                .with_bias(false)
                .init(device),
            hidden_out: LinearConfig::new(self.d_model, self.n_labels)
                .with_bias(true)
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct Code2Vec<B: Backend> {
    /// Embeddings of all terminals within the corpus.
    terminals: Embedding<B>,
    /// Embeddings of all paths within the corpus.
    paths: Embedding<B>,
    /// The attention weights for code2vec.
    attention: Tensor<B, 2, Float>,
    /// The hidden input layer.
    hidden_in: Linear<B>,
    /// The hidden output layer.
    hidden_out: Linear<B>,
}

impl<B: Backend> Code2Vec<B> {
    pub fn forward(
        &self,
        start_terminals: Tensor<B, 2, Int>,
        end_terminals: Tensor<B, 2, Int>,
        paths: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2, Float> {
        let start_emb = self.terminals.forward(start_terminals);
        let end_emb = self.terminals.forward(end_terminals);
        let path_emb = self.paths.forward(paths);
        println!("path_emb: {}", path_emb);

        let concat = Tensor::cat(vec![start_emb, path_emb, end_emb], 2);
        println!("concat: {}", concat);

        let num_context_vectors = concat.shape().dims[1];

        let hidden = self.hidden_in.forward(concat);

        let attn = self
            .attention
            .clone()
            .repeat_dim(0, num_context_vectors)
            .unsqueeze_dim(0);

        let hidden = hidden.mul(attn);

        // let attn_scores = hidden.matmul(&self.attention).softmax(0);
        // let attn_applied = hidden * attn_scores.unsqueeze(1);
        // let code_vec = attn_applied.sum_dim(0);

        // self.hidden_out.forward(code_vec.unsqueeze(0))
        // todo!()
        hidden.squeeze(1)
    }
}

#[test]
fn test_forward() {
    use burn::backend::NdArray;

    let dev = Default::default();
    let model = Code2VecConfig::new(10, 50, 30, 10, 10, 10).init::<NdArray>(&dev);

    let out = model.forward(
        Tensor::from_data([[0, 1, 2]], &dev),
        Tensor::from_data([[3, 4, 5]], &dev),
        Tensor::from_data([[15, 16, 17]], &dev),
    );

    println!("{}", out);
}
