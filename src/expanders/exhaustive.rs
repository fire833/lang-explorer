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

use crate::{
    errors::LangExplorerError,
    expanders::GrammarExpander,
    grammar::{
        grammar::Grammar, lhs::ProductionLHS, prod::Production, rule::ProductionRule, NonTerminal,
        Terminal,
    },
};

pub struct ExhaustiveExpander {}

impl<T: Terminal, I: NonTerminal> GrammarExpander<T, I> for ExhaustiveExpander {
    /// We may need to initialize the expander depending on the type of grammar
    /// we are using. For example, with my ML based example, the internal models of
    /// the expander may change completely depending on the rules of the grammar
    /// I want to expand.
    fn init(_grammar: &Grammar<T, I>) -> Result<Self, LangExplorerError>
    where
        Self: Sized,
    {
        todo!()
    }

    /// We want the expander to take a grammar and the current rule and
    /// make a decision on what the next expansion should be.
    fn expand_rule<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        _production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        todo!()
    }

    /// For context sensitive grammars, we could be in a situation where we have
    /// multiple left-hand sides that match some point on the frontier, along with
    /// multiple positions within the frontier where we could expand such left-hand side
    /// with a production. Thus, we want the expander to have the ability to make this
    /// decision on our behalf as well.
    fn choose_lhs_and_slot<'a>(
        &mut self,
        _grammar: &'a Grammar<T, I>,
        _lhs_location_matrix: &Vec<(&'a ProductionLHS<T, I>, Vec<usize>)>,
    ) -> (&'a ProductionLHS<T, I>, usize) {
        todo!()
    }
}
