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
    module::Module,
    nn::{Embedding, Linear},
    prelude::Backend,
};

use crate::{
    errors::LangExplorerError,
    expanders::GrammarExpander,
    grammar::{Grammar, NonTerminal, Production, ProductionRule, Terminal},
};

pub struct LearnedExpander<T, I, B>
where
    T: Terminal,
    I: NonTerminal,
    B: Backend,
{
    production_to_model: HashMap<Production<T, I>, ModuleWrapper<B>>,
}

enum ModuleWrapper<B>
where
    B: Backend,
{
    Linear(Linear<B>),
    Embedding(Embedding<B>),
}

impl<T, I, B> GrammarExpander<T, I> for LearnedExpander<T, I, B>
where
    T: Terminal,
    I: NonTerminal,
    B: Backend,
{
    fn init<'a>(grammar: &'a Grammar<T, I>) -> Result<Self, LangExplorerError> {
        Ok(Self {
            production_to_model: HashMap::new(),
        })
    }

    fn expand_rule<'a>(
        &mut self,
        grammar: &'a Grammar<T, I>,
        production: &'a Production<T, I>,
    ) -> &'a ProductionRule<T, I> {
        todo!()
    }
}
