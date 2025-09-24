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

use core::panic;

use burn::{optim::AdamWConfig, tensor::backend::AutodiffBackend};

use crate::{
    embedding::{
        doc2vecdbowns::{Doc2VecDBOWNSEmbedderParams, Doc2VecEmbedderDBOWNS},
        GeneralEmbeddingTrainingParams, LanguageEmbedder,
    },
    errors::LangExplorerError,
    expanders::{EmbedderWrapper, GrammarExpander},
    grammar::{
        grammar::Grammar,
        lhs::ProductionLHS,
        prod::Production,
        program::{ProgramInstance, WLKernelHashingOrder},
        rule::ProductionRule,
        NonTerminal, Terminal,
    },
    tooling::{modules::embed::AggregationMethod, training::TrainingParams},
};

pub struct LearnedExpander<T: Terminal, I: NonTerminal, B: AutodiffBackend> {
    /// The embedder to create embeddings for a
    /// new program or partial program.
    embedder: EmbedderWrapper<T, I, B>,

    /// Number of iterations to run to extract words.
    wl_kernel_iterations: u32,
    /// The order in which to hash items when computing new labels.
    wl_kernel_order: WLKernelHashingOrder,
    /// Toggle whether to deduplicate words when extracting WL kernel features.
    wl_dedup: bool,
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> GrammarExpander<T, I>
    for LearnedExpander<T, I, B>
{
    fn init(grammar: &Grammar<T, I>, _seed: u64) -> Result<Self, LangExplorerError> {
        let device = Default::default();

        let d2v = Doc2VecEmbedderDBOWNS::new(
            grammar,
            Doc2VecDBOWNSEmbedderParams::new(
                AdamWConfig::new(),
                1000,
                1000,
                GeneralEmbeddingTrainingParams::new(
                    AggregationMethod::Average,
                    TrainingParams::new(),
                ),
                "".to_string(),
            ),
            device,
        );

        Ok(Self {
            embedder: EmbedderWrapper::Doc2Vec(d2v),
            wl_kernel_iterations: 5,
            wl_kernel_order: WLKernelHashingOrder::ParentSelfChildrenOrdered,
            wl_dedup: true,
        })
    }

    fn expand_rule<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        context: &'a ProgramInstance<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        let doc = context.clone();
        let words = doc.extract_words_wl_kernel(
            self.wl_kernel_iterations,
            self.wl_kernel_order.clone(),
            self.wl_dedup,
            false,
        );

        let _embedding = match self.embedder.forward(doc, words) {
            Ok(e) => e,
            Err(err) => panic!("{}", err),
        };

        production.get(0).expect("couldn't find the selected index")
    }

    /// For context sensitive grammars, we could be in a situation where we have
    /// multiple left-hand sides that match some point on the frontier, along with
    /// multiple positions within the frontier where we could expand such left-hand side
    /// with a production. Thus, we want the expander to have the ability to make this
    /// decision on our behalf as well.
    fn choose_lhs_and_slot<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        _context: &'a ProgramInstance<T, I>,
        _lhs_location_matrix: &[(&'a ProductionLHS<T, I>, Vec<usize>)],
    ) -> (&'a ProductionLHS<T, I>, usize) {
        todo!()
    }

    /// Whenever a program has finished being generated, this method will be called
    /// to reset/update internal state in the expander. This is mostly going to be used
    /// in the learned expander to run backprop and update the internal models for
    /// generating the program in the first place.
    fn cleanup(&mut self) {
        todo!()
    }
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> LearnedExpander<T, I, B> {}
