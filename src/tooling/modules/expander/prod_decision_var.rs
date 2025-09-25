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

use std::collections::HashMap;

use burn::{
    config::Config,
    module::{Ignored, Module},
    prelude::Backend,
    tensor::{
        activation::{log_softmax, softmax},
        Float, Int, Tensor,
    },
};

use crate::{
    expanders::learned::{NormalizationStrategy, SamplingStrategy},
    grammar::{grammar::Grammar, prod::Production, NonTerminal, Terminal},
    tooling::modules::expander::{
        lin2::Linear2DeepConfig, lin3::Linear3DeepConfig, lin4::Linear4DeepConfig,
        LinearModuleWrapper, ProductionModelType,
    },
};

#[derive(Debug, Config)]
pub struct ProductionDecisionVariableConfig {}

impl ProductionDecisionVariableConfig {
    pub fn init<T: Terminal, I: NonTerminal, B: Backend>(
        &self,
        grammar: &Grammar<T, I>,
        device: &B::Device,
    ) -> ProductionDecisionVariable<B> {
        let mut map = HashMap::new();
        let mut items = vec![];

        for (idx, production) in grammar.get_productions().iter().enumerate() {
            let module = match production.ml_config.model {
                ProductionModelType::Linear2 => LinearModuleWrapper::Linear2(
                    Linear2DeepConfig::new(production.len())
                        .with_bias(production.ml_config.with_bias)
                        .init::<B>(&device),
                ),
                ProductionModelType::Linear3 => LinearModuleWrapper::Linear3(
                    Linear3DeepConfig::new(production.len())
                        .with_bias(production.ml_config.with_bias)
                        .init::<B>(&device),
                ),
                ProductionModelType::Linear4 => LinearModuleWrapper::Linear4(
                    Linear4DeepConfig::new(production.len())
                        .with_bias(production.ml_config.with_bias)
                        .init::<B>(&device),
                ),
            };

            items.push(module);
            let prod = production.hash_internal();
            map.insert(prod, idx);
        }

        ProductionDecisionVariable {
            models: items,
            production_to_model: Ignored(map),
        }
    }
}

#[derive(Debug, Module)]
pub struct ProductionDecisionVariable<B: Backend> {
    /// Do this instead of hashing the production directly because of some
    /// weird typing bugs with burn. Idk.
    production_to_model: Ignored<HashMap<u64, usize>>,

    /// Mapping of each production rule to its corresponding decision function.
    models: Vec<LinearModuleWrapper<B>>,
}

impl<B: Backend> ProductionDecisionVariable<B> {
    pub fn forward<'a, T: Terminal, I: NonTerminal>(
        &self,
        productions: Vec<&'a Production<T, I>>,
        inputs: Tensor<B, 2, Float>,
    ) -> Tensor<B, 2, Int> {
        let mut outputs: Vec<Tensor<B, 1, Int>> = vec![];

        for (idx, prod) in productions.iter().enumerate() {
            let model = self
                .models
                .get(
                    self.production_to_model
                        .0
                        .get(&prod.hash_internal())
                        .cloned()
                        .unwrap(),
                )
                .unwrap_or_else(|| panic!("could not find model for {:?}", prod));

            let emb = inputs
                .clone()
                .gather(1, Tensor::from_data([0, 1], &inputs.device()))
                .unsqueeze();

            let logits = match model {
                LinearModuleWrapper::Linear2(linear) => {
                    linear.forward(emb, prod.ml_config.activation.clone())
                }
                LinearModuleWrapper::Linear3(linear) => {
                    linear.forward(emb, prod.ml_config.activation.clone())
                }
                LinearModuleWrapper::Linear4(linear) => {
                    linear.forward(emb, prod.ml_config.activation.clone())
                }
            };

            // Optionally scale by temperature.
            let logits = if prod.ml_config.temperature != 1000 {
                logits.div_scalar(prod.ml_config.temperature as f32 * 0.001)
            } else {
                logits
            };

            let output = match prod.ml_config.normalization {
                NormalizationStrategy::Softmax => softmax(logits, 1).squeeze::<1>(0),
                NormalizationStrategy::LogSoftmax => log_softmax(logits, 1).squeeze::<1>(0),
            };

            let loss = match prod.ml_config.sampling {
                SamplingStrategy::HighestProb => output.argmax(0),
                SamplingStrategy::LowestProb => output.argmin(0),
            };

            outputs.push(loss);
        }

        Tensor::stack(outputs, 0)
    }
}

#[test]
fn test_forward() {}
