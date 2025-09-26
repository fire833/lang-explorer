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

use burn::tensor::backend::AutodiffBackend;

use crate::{
    errors::LangExplorerError,
    expanders::{
        EmbedderWrapper, FrontierDecisionWrapper, GrammarExpander, ProductionDecisionWrapper,
    },
    grammar::{
        grammar::Grammar, lhs::ProductionLHS, prod::Production, program::ProgramInstance,
        rule::ProductionRule, NonTerminal, Terminal,
    },
};

/// The different strategies for choosing the next expansion rule
/// given the probability distribution from the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Choose the highest probability expansion.
    HighestProb,

    /// Choose the lowest probability distribution,
    /// you probably don't want to do this if you
    /// care about your output.
    LowestProb,
}

/// The different strategies for normalizing the output logits
/// given by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationStrategy {
    Softmax,
    LogSoftmax,
}

pub struct LearnedExpander<T: Terminal, I: NonTerminal, B: AutodiffBackend> {
    /// Model for embedding partial programs.
    embedder: EmbedderWrapper<T, I, B>,

    /// Wrapper for production decision model.
    production_decision: ProductionDecisionWrapper<B>,

    /// Wrapper for frontier decision model.
    frontier_decision: FrontierDecisionWrapper<B>,
}

impl<T: Terminal, I: NonTerminal, B: AutodiffBackend> GrammarExpander<T, I>
    for LearnedExpander<T, I, B>
{
    fn init(grammar: &Grammar<T, I>, seed: u64) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        B::seed(seed);

        todo!()
    }

    fn expand_rule<'a>(
        &mut self,
        grammar: &'a Grammar<T, I>,
        context: &'a ProgramInstance<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        todo!()
    }

    fn choose_lhs_and_slot<'a>(
        &mut self,
        grammar: &'a Grammar<T, I>,
        context: &'a ProgramInstance<T, I>,
        lhs_location_matrix: &[(&'a ProductionLHS<T, I>, Vec<usize>)],
    ) -> (&'a ProductionLHS<T, I>, usize) {
        todo!()
    }

    fn cleanup(&mut self) {
        todo!()
    }
}
