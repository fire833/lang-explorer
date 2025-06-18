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

use crate::{
    errors::LangExplorerError,
    grammar::{
        elem::GrammarElement, lhs::ProductionLHS, prod::context_free_production,
        prod::production_rule, prod::Production, rule::ProductionRule, Grammar,
    },
    languages::{
        strings::{
            alphanumeric::{
                alpha_character_production_context_free,
                alpha_lower_character_production_context_free,
            },
            nterminal_str, StringValue, COMMA, EQUALS, LPAREN, RPAREN, STAR,
        },
        GrammarBuilder,
    },
};

nterminal_str!(NT_ENTRY, "entrypoint");
nterminal_str!(NT_EXPR, "expression");
nterminal_str!(NT_ELEMENT, "element");
nterminal_str!(SYMBOL, "symbol");
nterminal_str!(NT_INDEX, "index");
nterminal_str!(INDEX, "index");

pub struct TacoExpressionLanguage;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TacoExpressionLanguageParams {}

impl GrammarBuilder for TacoExpressionLanguage {
    type Term = StringValue;
    type NTerm = StringValue;
    type Params<'de> = TacoExpressionLanguageParams;

    fn generate_grammar<'de>(
        _params: Self::Params<'de>,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError> {
        let grammar = Grammar::new(
            "entrypoint".into(),
            vec![
                context_free_production!(NT_ENTRY, production_rule!(NT_ELEMENT, EQUALS, NT_EXPR)),
                context_free_production!(
                    NT_ELEMENT,
                    production_rule!(SYMBOL, LPAREN, NT_INDEX, RPAREN)
                ),
                context_free_production!(
                    NT_INDEX,
                    production_rule!(NT_INDEX),
                    production_rule!(NT_INDEX, COMMA, NT_INDEX),
                    production_rule!(INDEX)
                ),
                context_free_production!(
                    NT_EXPR,
                    production_rule!(NT_EXPR, STAR, NT_EXPR),
                    production_rule!(NT_ELEMENT)
                ),
                alpha_lower_character_production_context_free("index".into()),
                alpha_character_production_context_free("symbol".into()),
            ],
        );

        Ok(grammar)
    }
}
