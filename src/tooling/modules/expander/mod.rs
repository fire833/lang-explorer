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

use crate::expanders::learned::{NormalizationStrategy, SamplingStrategy};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductionModelType {
    Linear1,
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
