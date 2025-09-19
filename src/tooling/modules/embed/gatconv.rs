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
    module::{Module, Param},
    nn::{Dropout, DropoutConfig},
    prelude::Backend,
    tensor::{Float, Tensor},
};

#[derive(Debug, Config)]
pub struct GATConvConfig {}

impl GATConvConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GATConv<B> {
        GATConv {
            attn_l: Param::from_tensor(Tensor::random(
                [4, 4],
                burn::tensor::Distribution::Uniform(-1., 1.),
                device,
            )),
            attn_r: Param::from_tensor(Tensor::random(
                [4, 4],
                burn::tensor::Distribution::Uniform(-1., 1.),
                device,
            )),
            feat_drop: DropoutConfig::new(0.1).init(),
            attn_drop: DropoutConfig::new(0.1).init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct GATConv<B: Backend> {
    attn_l: Param<Tensor<B, 2>>,
    attn_r: Param<Tensor<B, 2>>,

    feat_drop: Dropout,
    attn_drop: Dropout,
}

impl<B: Backend> GATConv<B> {
    pub fn forward(&self) -> Tensor<B, 2, Float> {
        todo!()
    }
}

#[test]
fn test_forward() {}
