/*
*	Copyright (C) 2024 Kendall Tauser
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

use async_trait::async_trait;

use crate::grammar::{Grammar, NonTerminal, Production, ProductionRule, Terminal};

/// A grammar expander is an object that is able to take a
/// current production rule, the whole of the grammar that is
/// being utilized, and is able to spit out a production rule
/// that should be utilized from the list of possible production
/// rules that are implemented by this production.
#[async_trait]
pub trait GrammarExpander<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn expand_rule(
        &self,
        grammar: &Grammar<T, I>,
        production: &Production<T, I>,
    ) -> &ProductionRule<T, I>;
}
