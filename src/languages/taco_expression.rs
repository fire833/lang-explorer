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

use std::collections::HashSet;

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
        NonTerminal,
    },
    languages::{
        strings::{nterminal_str, COMMA, EQUALS, FORWARDSLASH, LPAREN, MINUS, PLUS, RPAREN, STAR},
        GrammarBuilder, GrammarState,
    },
};

nterminal_str!(NT_ENTRY, "entrypoint");
nterminal_str!(NT_EXPR, "expr");
nterminal_str!(NT_ELEMENT, "elem");
nterminal_str!(TENSOR, "symbol");
nterminal_str!(NT_INDEX, "nt_index");
nterminal_str!(INDEX, "index");

pub struct TacoExpressionLanguage;

#[derive(Default, Debug)]
pub struct TacoExpressionState {
    symbols: HashSet<GrammarElement>,
    in_tensor: bool,
}

impl GrammarState for TacoExpressionState {
    fn apply_context(&mut self, prod: &Production) -> Option<Production> {
        const SYMB: NonTerminal = NonTerminal::ConstStr("symbol");
        if prod.non_terminal.non_terminal == SYMB {
            let mut prod = prod.clone();
            // filter out production rules that are not in symbols
            prod.items
                .retain(|rule| !self.symbols.contains(&rule.items[0]));

            self.in_tensor = true;
            return Some(prod);
        }

        None
    }

    fn update(&mut self, rule: &ProductionRule) {
        if self.in_tensor {
            self.in_tensor = false;
            self.symbols.insert(rule.items[0].clone());
        }
    }
}

/// Parameters for Taco Expression Language.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TacoExpressionLanguageParams {
    #[serde(alias = "version")]
    version: TacoExpressionLanguageVersion,

    symbols: Vec<char>,
    indices: Vec<char>,
}

impl Default for TacoExpressionLanguageParams {
    fn default() -> Self {
        Self {
            version: Default::default(),
            symbols: ('A'..='Z').collect(),
            indices: ('a'..='z').collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum TacoExpressionLanguageVersion {
    /// Generate expressions with the context free version of this grammar.
    /// Please note, this version can lead to syntactically incorrect programs.
    ContextFreeV1,
    ContextSensitiveV1,
}

impl Default for TacoExpressionLanguageVersion {
    fn default() -> Self {
        Self::ContextFreeV1
    }
}

impl GrammarBuilder for TacoExpressionLanguage {
    type Params<'de> = TacoExpressionLanguageParams;
    type State = TacoExpressionState;

    fn generate_grammar<'de>(params: Self::Params<'de>) -> Result<Grammar, LangExplorerError> {
        let mut symbols: Vec<ProductionRule> = vec![];
        for var in params.symbols.iter() {
            // Store this variable in the heap.
            let term = GrammarElement::Terminal(String::from(*var).into());
            symbols.push(production_rule!(term));
        }

        let mut indices: Vec<ProductionRule> = vec![];
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
                        production_rule!(TENSOR, LPAREN, NT_INDEX, RPAREN)
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
                    Production::new(ProductionLHS::new_context_free_elem(TENSOR), symbols),
                ],
                "tacoexpr".into(),
            ),
            TacoExpressionLanguageVersion::ContextSensitiveV1 => todo!(),
        };

        Ok(grammar)
    }
}
