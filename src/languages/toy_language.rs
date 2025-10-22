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

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    errors::LangExplorerError,
    grammar::{
        elem::GrammarElement,
        grammar::Grammar,
        lhs::ProductionLHS,
        prod::{production_rule, Production},
        rule::ProductionRule,
        NonTerminal, Terminal,
    },
    languages::GrammarMutator,
};

use super::{
    strings::{nterminal_str, terminal_str, StringValue, EPSILON},
    GrammarBuilder,
};

nterminal_str!(S, "S");
terminal_str!(A, "a");
terminal_str!(B, "b");

pub struct ToyLanguage;

pub struct ToyLanguageChecker;
impl<T: Terminal, I: NonTerminal> GrammarMutator<T, I> for ToyLanguageChecker {}

#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema)]
pub struct ToyLanguageParams {}

impl GrammarBuilder for ToyLanguage {
    type Term = StringValue;
    type NTerm = StringValue;
    type Params<'de> = ToyLanguageParams;
    type Mutator = ToyLanguageChecker;

    fn generate_grammar<'de>(
        _params: Self::Params<'de>,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError> {
        Ok(Grammar::new(
            "S".into(),
            vec![Production::new(
                ProductionLHS::new_context_free_elem(S),
                vec![
                    production_rule!(S, S),
                    production_rule!(A, A, S, B, B),
                    production_rule!(EPSILON),
                ],
            )],
            "toy".into(),
        ))
    }

    fn new_mutator() -> Self::Mutator {
        ToyLanguageChecker {}
    }
}
