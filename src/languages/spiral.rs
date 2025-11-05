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
        NonTerminal, Terminal,
    },
    languages::{
        strings::{nterminal_str, terminal_str, StringValue, LPAREN, RPAREN},
        GrammarBuilder, GrammarState,
    },
};

nterminal_str!(SPL, "spl");
nterminal_str!(GENERIC, "generic");
nterminal_str!(SYMBOL, "symbol");
nterminal_str!(TRANSFORM, "transform");
nterminal_str!(OTIMESSPL, "otimesspl");
nterminal_str!(OPLUSSPL, "oplusspl");
nterminal_str!(DFT, "dft");
nterminal_str!(WHT, "wht");
terminal_str!(OTIMES, "\\otimes");
terminal_str!(OPLUS, "\\oplus");
terminal_str!(I_N, "I_n");
terminal_str!(I_M, "I_m");
terminal_str!(I_K, "I_k");
terminal_str!(J_N, "J_n");
terminal_str!(P_N, "P_n");
terminal_str!(Q_N, "Q_n");
terminal_str!(F2, "F_2");
terminal_str!(L_N_K, "L_{n,k}");
terminal_str!(T_N_M, "T_{n,m}");
terminal_str!(R_ALPHA, "R_\\alpha");

pub struct SpiralLanguage;

pub struct SpiralState {}

impl<T: Terminal, I: NonTerminal> GrammarState<T, I> for SpiralState {
    fn apply_context<'a>(&mut self, prod: &'a Production<T, I>) -> Option<Production<T, I>> {
        todo!()
    }

    fn update(&mut self, rule: &ProductionRule<T, I>) {
        todo!()
    }
}

impl Default for SpiralState {
    fn default() -> Self {
        SpiralState {}
    }
}

/// Parameters for SPIRAL Language.
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema)]
pub struct SpiralLanguageParams {}

impl GrammarBuilder for SpiralLanguage {
    type Term = StringValue;
    type NTerm = StringValue;
    type Params<'de> = SpiralLanguageParams;
    type State = SpiralState;

    fn generate_grammar<'de>(
        _params: Self::Params<'de>,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError> {
        let grammar = Grammar::new(
            "spl".into(),
            vec![
                context_free_production!(
                    SPL,
                    // production_rule!(GENERIC),
                    production_rule!(SYMBOL),
                    production_rule!(TRANSFORM),
                    production_rule!(OTIMESSPL, OTIMES, OTIMESSPL),
                    production_rule!(OPLUSSPL, OPLUS, OPLUSSPL)
                ),
                context_free_production!(
                    OPLUSSPL,
                    // production_rule!(GENERIC),
                    production_rule!(SYMBOL),
                    production_rule!(TRANSFORM),
                    production_rule!(OPLUSSPL, OPLUS, OPLUSSPL)
                ),
                context_free_production!(
                    OTIMESSPL,
                    // production_rule!(GENERIC),
                    production_rule!(SYMBOL),
                    production_rule!(TRANSFORM),
                    production_rule!(OTIMESSPL, OTIMES, OTIMESSPL)
                ),
                context_free_production!(
                    SYMBOL,
                    production_rule!(I_N),
                    production_rule!(J_N),
                    production_rule!(F2),
                    production_rule!(L_N_K),
                    production_rule!(R_ALPHA)
                ),
                context_free_production!(TRANSFORM, production_rule!(DFT), production_rule!(WHT)),
                // context_free_production!(GENERIC),
                context_free_production!(
                    DFT,
                    production_rule!(
                        LPAREN, DFT, OTIMES, I_M, RPAREN, T_N_M, LPAREN, I_K, OTIMES, DFT, RPAREN,
                        L_N_K
                    ),
                    production_rule!(P_N, LPAREN, DFT, OTIMES, DFT, RPAREN, Q_N),
                    production_rule!(F2)
                ),
                context_free_production!(
                    WHT,
                    production_rule!(F2),
                    production_rule!(F2, OTIMES, WHT)
                ),
            ],
            "spiral".into(),
        );

        Ok(grammar)
    }

    fn new_state() -> Option<Self::State> {
        Some(Default::default())
    }
}
