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
    tensor::{activation::sigmoid, s, Distribution, Float, Int, Shape, Tensor},
};

#[allow(unused)]
use burn::backend::NdArray;

#[derive(Debug, Config)]
pub struct NegativeSamplingConfig {}

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
    /// - word_indices: `[batch_size, k + 1]`
    /// - input: `[batch_size, n_words]`
    /// - output: `[1]`
    pub fn forward(
        &self,
        word_indices: Tensor<B, 2, Int>,
        input: Tensor<B, 2, Float>,
    ) -> Tensor<B, 1, Float> {
        let batch_size = input.shape().dims::<2>()[0];
        let log_sigmoid = sigmoid(input.clone().slice(s![..])).log();
        let positive = sigmoid(input.slice(s![..])).log();
        return -positive
            // .add(-negatives)
            .div_scalar(batch_size as u32)
            .squeeze(1);
    }
}

#[test]
fn test_forward() {
    let dev = Default::default();
    let model = NegativeSamplingConfig::new().init::<NdArray>(&dev);

    // let res = model.forward(Tensor::from_data(
    //     [[1.00, 2.00, 3.00], [4.00, 5.00, 6.00]],
    //     &dev,
    // ));

    // println!("res: {res}");
}
