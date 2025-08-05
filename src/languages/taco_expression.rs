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
        prod::{context_free_production, production_rule, Production},
        rule::ProductionRule,
    },
    languages::{
        strings::{
            nterminal_str, StringValue, COMMA, EQUALS, FORWARDSLASH, LPAREN, MINUS, PLUS, RPAREN,
            STAR,
        },
        GrammarBuilder,
    },
};

nterminal_str!(NT_ENTRY, "entrypoint");
nterminal_str!(NT_EXPR, "expression");
nterminal_str!(NT_ELEMENT, "element");
nterminal_str!(SYMBOL, "symbol");
nterminal_str!(NT_INDEX, "nt_index");
nterminal_str!(INDEX, "index");

pub struct TacoExpressionLanguage;

/// Parameters for Taco Expression Language.
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema)]
pub struct TacoExpressionLanguageParams {
    #[serde(alias = "version")]
    version: TacoExpressionLanguageVersion,

    symbols: Vec<char>,
    indices: Vec<char>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub enum TacoExpressionLanguageVersion {
    ContextFreeV1,
}

impl Default for TacoExpressionLanguageVersion {
    fn default() -> Self {
        Self::ContextFreeV1
    }
}

impl GrammarBuilder for TacoExpressionLanguage {
    type Term = StringValue;
    type NTerm = StringValue;
    type Params<'de> = TacoExpressionLanguageParams;

    fn generate_grammar<'de>(
        params: Self::Params<'de>,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError> {
        let mut symbols: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        for var in params.symbols.iter() {
            // Store this variable in the heap.
            let term = GrammarElement::Terminal(String::from(*var).into());
            symbols.push(production_rule!(term));
        }

        let mut indices: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        for var in params.indices.iter() {
            let term = GrammarElement::Terminal(String::from(*var).into());
            indices.push(production_rule!(term));
        }

        let grammar = match params.version {
            TacoExpressionLanguageVersion::ContextFreeV1 => Grammar::new(
                "entrypoint".into(),
                vec![
                    context_free_production!(
                        NT_ENTRY,
                        production_rule!(NT_ELEMENT, EQUALS, NT_EXPR)
                    ),
                    context_free_production!(
                        NT_ELEMENT,
                        production_rule!(SYMBOL, LPAREN, NT_INDEX, RPAREN)
                    ),
                    context_free_production!(
                        NT_INDEX,
                        production_rule!(30, NT_INDEX, COMMA, NT_INDEX),
                        production_rule!(70, INDEX)
                    ),
                    context_free_production!(
                        NT_EXPR,
                        production_rule!(10, NT_EXPR, STAR, NT_EXPR),
                        production_rule!(10, NT_EXPR, PLUS, NT_EXPR),
                        production_rule!(10, NT_EXPR, MINUS, NT_EXPR),
                        production_rule!(10, NT_EXPR, FORWARDSLASH, NT_EXPR),
                        production_rule!(50, NT_ELEMENT)
                    ),
                    // Fused index variable
                    Production::new(ProductionLHS::new_context_free_elem(INDEX), indices),
                    Production::new(ProductionLHS::new_context_free_elem(SYMBOL), symbols),
                ],
                "tacoexpr".into(),
            ),
        };

        Ok(grammar)
    }
}
