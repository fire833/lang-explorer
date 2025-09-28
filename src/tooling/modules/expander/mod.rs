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
    module::Module,
    prelude::Backend,
    tensor::{Float, Tensor},
};

use crate::{
    expanders::learned::{NormalizationStrategy, SamplingStrategy},
    tooling::modules::expander::{
        lin1::Linear1Deep, lin2::Linear2Deep, lin3::Linear3Deep, lin4::Linear4Deep,
    },
};

pub mod lin1;
pub mod lin2;
pub mod lin3;
pub mod lin4;

pub mod frontier_decision;
pub mod prod_decision_attn;
pub mod prod_decision_fixed;
pub mod prod_decision_var;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Activation {
    Sigmoid,
    ReLU,
    LeakyReLU,
    TanH,
}

/// A bit of a hack to allow us to keep an enum for all linear model types.
#[derive(Module, Debug)]
pub enum LinearModuleWrapper<B: Backend> {
    Linear1(Linear1Deep<B>),
    Linear2(Linear2Deep<B>),
    Linear3(Linear3Deep<B>),
    Linear4(Linear4Deep<B>),
}

impl<B: Backend> LinearModuleWrapper<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 2, Float>,
        activation: Activation,
    ) -> Tensor<B, 2, Float> {
        match self {
            LinearModuleWrapper::Linear1(model) => model.forward(input, activation.clone()),
            LinearModuleWrapper::Linear2(model) => model.forward(input, activation.clone()),
            LinearModuleWrapper::Linear3(model) => model.forward(input, activation.clone()),
            LinearModuleWrapper::Linear4(model) => model.forward(input, activation.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductionModelType {
    Linear2,
    Linear3,
    Linear4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionConfiguration {
    sampling: SamplingStrategy,
    normalization: NormalizationStrategy,
    activation: Activation,
    with_bias: bool,
    model: ProductionModelType,
    // Temperature value that will be divided by 1000 to get actual temperature.
    temperature: u32,
}

impl ProductionConfiguration {
    pub const fn new() -> Self {
        Self {
            sampling: SamplingStrategy::HighestProb,
            normalization: NormalizationStrategy::LogSoftmax,
            activation: Activation::ReLU,
            model: ProductionModelType::Linear2,
            with_bias: true,
            temperature: 1000,
        }
    }
}

impl Default for ProductionConfiguration {
    fn default() -> Self {
        Self::new()
    }
}
