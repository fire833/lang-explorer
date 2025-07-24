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

use std::marker::PhantomData;

use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
    tensor::{activation::sigmoid, Float, Int, Tensor},
};

#[allow(unused)]
use burn::tensor::Distribution;

#[allow(unused)]
use burn::backend::NdArray;

#[derive(Debug, Config)]
pub struct NegativeSamplingConfig;

impl NegativeSamplingConfig {
    pub fn init<B: Backend>(&self, _device: &B::Device) -> NegativeSampling<B> {
        NegativeSampling::<B> { p: PhantomData {} }
    }
}

#[derive(Debug, Module)]
pub struct NegativeSampling<B: Backend> {
    p: PhantomData<B>,
}

impl<B: Backend> NegativeSampling<B> {
    /// Applies the forward pass to the input tensor.
    ///
    /// Specifically, takes the input logits and the indices of ground truth words
    /// and returns the loss. k denotes the number of negative samples that will be
    /// updated on any given backward pass. For computational efficiency, the positive index
    /// should be the 0th index for every batch item, because the log sigmoid value is
    /// computed on the entire tensor at once for speed.
    ///
    /// # Shapes
    ///
    /// - true_word_indices: `[batch_size]`
    /// - word_indices: `[batch_size, k]`
    /// - input: `[batch_size, n_words]`
    /// - output: `[1]`
    pub fn forward(
        &self,
        true_word_indices: Tensor<B, 1, Int>,
        word_indices: Tensor<B, 2, Int>,
        input: Tensor<B, 2, Float>,
    ) -> Tensor<B, 1, Float> {
        // TODO: this clone will be VERY expensive, need to figure out a way to get rid of this.
        let negatives = sigmoid(input.clone().gather(1, word_indices))
            .log()
            .sum_dim(1)
            .squeeze(1);
        let positive = sigmoid(input.gather(1, true_word_indices.unsqueeze_dim(1)))
            .log()
            .squeeze(1);
        return -positive.add(-negatives);
    }
}

#[test]
fn test_forward() {
    let dev = Default::default();
    let model = NegativeSamplingConfig::new().init::<NdArray>(&dev);

    let input = Tensor::<NdArray, 2>::random([5, 24], Distribution::Default, &dev);
    let words = Tensor::<NdArray, 2, Int>::from_data(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [10, 11, 12, 13, 14],
            [12, 13, 14, 15, 16],
            [15, 16, 17, 18, 19],
        ],
        &dev,
    );
    let true_words = Tensor::<NdArray, 1, Int>::from_data([23, 23, 23, 23, 23], &dev);

    println!("inputs: {input}");
    println!("words: {words}");

    let res = model.forward(true_words, words, input);

    println!("res: {res}");
}
