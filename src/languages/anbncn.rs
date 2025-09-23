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

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    errors::LangExplorerError,
    grammar::{
        elem::GrammarElement,
        grammar::Grammar,
        lhs::ProductionLHS,
        prod::{
            context_free_production, production_rule, single_prefix_production,
            single_suffix_production, Production,
        },
        rule::ProductionRule,
        NonTerminal, Terminal,
    },
    languages::{
        strings::{alphanumeric::T_LO_A, nterminal_str, StringValue},
        GrammarBuilder, GrammarExpansionChecker,
    },
};

nterminal_str!(S, "S");
nterminal_str!(A, "A");
nterminal_str!(B, "B");
nterminal_str!(C, "C");
nterminal_str!(W, "W");
nterminal_str!(Z, "Z");

pub struct AnBnCnLanguage {}

pub struct AnBnCnLanguageChecker;
impl<T: Terminal, I: NonTerminal> GrammarExpansionChecker<T, I> for AnBnCnLanguageChecker {}

/// Parameters for AnBnCn language.
#[derive(Default, Debug, Serialize, Deserialize, ToSchema)]
pub struct AnBnCnLanguageParams {}

impl GrammarBuilder for AnBnCnLanguage {
    type Term = StringValue;
    type NTerm = StringValue;
    type Params<'de> = AnBnCnLanguageParams;
    type Checker = AnBnCnLanguageChecker;

    fn generate_grammar<'de>(
        _params: Self::Params<'de>,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError> {
        let g = Grammar::new(
            "S".into(),
            vec![
                context_free_production!(
                    S,
                    production_rule!(T_LO_A, B, C),
                    production_rule!(T_LO_A, S, B, C)
                ),
                single_suffix_production!("C".into(), B, production_rule!(Z)),
                Production::new(
                    ProductionLHS::new_with_prefix_list(vec![C], "B".into()),
                    vec![production_rule!(Z)],
                ),
                Production::new(
                    ProductionLHS::new_with_suffix_single(C, "Z".into()),
                    vec![production_rule!(W, Z)],
                ),
                Production::new(
                    ProductionLHS::new_with_prefix_list(vec![W], "Z".into()),
                    vec![production_rule!(W, C)],
                ),
            ],
            "anbncn".into(),
        );

        Ok(g)
    }

    fn new_checker() -> Self::Checker {
        AnBnCnLanguageChecker {}
    }
}
