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
    tensor::{activation::sigmoid, s, Float, Tensor},
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
    pub fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 1, Float> {
        let batch_size = input.shape().dims::<2>()[0];
        // let negative_samples = input.shape().dims()[1] - 1;
        let log_sigmoid = sigmoid(input).log();
        let negatives = log_sigmoid.clone().slice(s![.., 1..=-1]);
        let negatives = negatives.sum_dim(1);
        let positive = log_sigmoid.slice(s![.., 0]);
        return -positive
            .add(-negatives)
            .div_scalar(batch_size as u32)
            .squeeze(1);
    }
}

#[test]
fn test_forward() {
    let dev = Default::default();
    let model = NegativeSamplingConfig::new().init::<NdArray>(&dev);

    let res = model.forward(Tensor::from_data(
        [[1.00, 2.00, 3.00], [4.00, 5.00, 6.00]],
        &dev,
    ));

    println!("res: {res}");
}
