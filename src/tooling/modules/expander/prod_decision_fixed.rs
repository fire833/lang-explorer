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
    nn::{Embedding, EmbeddingConfig},
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
        lin2::{Linear2Deep, Linear2DeepConfig},
        Activation,
    },
};

#[derive(Debug, Config)]
pub struct ProductionDecisionFixedConfig {
    /// The number of dimensions of embeddings.
    d_model: usize,

    /// The dimension of input vectors.
    d_embed: usize,
}

impl ProductionDecisionFixedConfig {
    pub fn init<T: Terminal, I: NonTerminal, B: Backend>(
        &self,
        grammar: &Grammar<T, I>,
        device: &B::Device,
    ) -> ProductionDecisionFixed<B> {
        let prods = grammar.get_productions();
        let mut map = HashMap::new();

        let mut rule_count = 0;

        for prod in prods.iter() {
            let phash = prod.hash_internal();
            for rule in prod.items.iter() {
                map.insert((phash, rule.hash_internal()), rule_count);
                rule_count += 1;
            }
        }

        ProductionDecisionFixed {
            rule_embeddings: EmbeddingConfig::new(rule_count + 1, self.d_model).init(device),
            linear: Linear2DeepConfig::new(self.d_model)
                .with_bias(true)
                .with_d_embed(self.d_embed)
                .init(device),
            production_to_index: Ignored(map),
            config: Ignored(self.clone()),
            num_embeddings: Ignored(rule_count),
        }
    }
}

#[derive(Debug, Module)]
pub struct ProductionDecisionFixed<B: Backend> {
    /// Map from production to index in the embedding matrix.
    production_to_index: Ignored<HashMap<(u64, u64), usize>>,

    /// Configuration for this module.
    config: Ignored<ProductionDecisionFixedConfig>,

    num_embeddings: Ignored<usize>,

    /// Embeddings for each rule that we want to expand.
    rule_embeddings: Embedding<B>,
    /// The linear decision function.
    linear: Linear2Deep<B>,
}

impl<B: Backend> ProductionDecisionFixed<B> {
    /// Applies the forward pass to the input tensors.
    ///
    /// More specifically, takes input embeddings for a context,
    /// passes them through a linear layer, then compares the output
    /// vector with vectors for all specified rules and returns a
    /// probability distribution of rules that should be chosen.
    ///
    /// # Shapes
    ///
    /// - rules: `[batch_size, k]`
    /// - inputs: `[batch_size, d_embedding]`
    /// - output: `[batch_size, k]`
    pub fn forward<'a, T: Terminal, I: NonTerminal>(
        &self,
        productions: Vec<&'a Production<T, I>>,
        inputs: Tensor<B, 2, Float>,
        activation: Activation,
        normalization: NormalizationStrategy,
        sampling: SamplingStrategy,
    ) -> Tensor<B, 2, Int> {
        let max = productions.iter().map(|p| p.len()).max().unwrap_or(0);

        let mut rules: Tensor<B, 2, Int> = Tensor::ones([productions.len(), max], &inputs.device());
        let mut rule_indices: Vec<usize> = vec![];

        for (idx, prod) in productions.iter().enumerate() {
            let phash = prod.hash_internal();
            for rule in prod.items.iter() {
                let rhash = rule.hash_internal();
                let idx = *self.production_to_index.0.get(&(phash, rhash)).unwrap();
                rule_indices.push(idx);
            }

            // Pad with dummy rules if necessary.
            for _ in 0..(max - prod.len()) {
                rule_indices.push(self.num_embeddings.0);
            }

            let t: Tensor<B, 2, Int> =
                Tensor::<B, 1, Int>::from_data(rule_indices.as_slice(), &inputs.device())
                    .unsqueeze_dim(0);

            rules = rules.slice_assign([idx..idx + 1, 0..max], t);
            rule_indices.clear();
        }

        let lout = self.linear.forward(inputs, activation);
        let rules = self.rule_embeddings.forward(rules);
        let rule_count = rules.dims()[1];

        // Compute the repeated output tensor for comparing against every rule vector.
        let out: Tensor<B, 3> = lout.unsqueeze_dim(1).repeat_dim(1, rule_count);

        // Compare vectors.
        let logits = rules.mul(out).sum_dim(2);

        // Compute softmax.
        let outputs = match normalization {
            NormalizationStrategy::Softmax => softmax(logits, 1).squeeze(2),
            NormalizationStrategy::LogSoftmax => log_softmax(logits, 1).squeeze(2),
        };

        let loss = match sampling {
            SamplingStrategy::HighestProb => outputs.argmax(1),
            SamplingStrategy::LowestProb => outputs.argmin(1),
        };

        loss
    }
}

#[test]
fn test_forward() {
    use crate::languages::{
        strings::StringValue, taco_schedule::TacoScheduleLanguage, GrammarBuilder,
    };
    use burn::backend::NdArray;

    let tacosched = TacoScheduleLanguage::generate_grammar(Default::default()).unwrap();

    let dev = Default::default();
    let model = ProductionDecisionFixedConfig::new(5, 5)
        .init::<StringValue, StringValue, NdArray>(&tacosched, &dev);

    let prods = tacosched.get_productions();
    let len = prods.len();

    let out = model.forward(
        tacosched.get_productions(),
        Tensor::<NdArray, 2, Float>::random(
            [len, 5],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev,
        ),
        Activation::ReLU,
        NormalizationStrategy::Softmax,
        SamplingStrategy::HighestProb,
    );
    println!("out: {out}");
}
