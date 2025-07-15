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
            alphanumeric::{T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9},
            nterminal_str, terminal_str, StringValue, COLON, LPAREN, RPAREN, SEMICOLON, SPACE,
        },
        GrammarBuilder,
    },
};

nterminal_str!(PROGRAM, "program");
nterminal_str!(STMT, "stmt");
nterminal_str!(ACTION, "action");
nterminal_str!(CONDITIONAL, "cond");
nterminal_str!(COUNT, "count");
terminal_str!(START, "def run(): ");
terminal_str!(WHILE, "while");
terminal_str!(REPEAT, "repeat");
terminal_str!(IF, "if");
terminal_str!(IFELSE, "ifelse");
terminal_str!(ELSE, "else");
terminal_str!(FRONT_CLEAR, "frontIsClear");
terminal_str!(LEFT_CLEAR, "leftIsClear");
terminal_str!(RIGHT_CLEAR, "rightIsClear");
terminal_str!(MARKERS_PRESENT, "markersPresent");
terminal_str!(NO_MARKERS_PRESENT, "noMarkersPresent");
terminal_str!(NOT, "not");
terminal_str!(MOVE, "move");
terminal_str!(TURN_RIGHT, "turnRight");
terminal_str!(TURN_LEFT, "turnLeft");
terminal_str!(PICK_MARKER, "pickMarker");
terminal_str!(PUT_MARKER, "putMarker");

pub struct KarelLanguage;

/// Parameters for Karel Language.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct KarelLanguageParameters {}

impl Default for KarelLanguageParameters {
    fn default() -> Self {
        Self {}
    }
}

impl GrammarBuilder for KarelLanguage {
    type Term = StringValue;

    type NTerm = StringValue;

    type Params<'de> = KarelLanguageParameters;

    fn generate_grammar<'de>(
        _params: Self::Params<'de>,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError> {
        Ok(Grammar::new(
            "program".into(),
            vec![
                context_free_production!(PROGRAM, production_rule!(START, STMT)),
                context_free_production!(
                    STMT,
                    // While loop
                    production_rule!(WHILE, LPAREN, CONDITIONAL, RPAREN, COLON, SPACE, STMT),
                    // Repeat loop
                    production_rule!(REPEAT, LPAREN, COUNT, RPAREN, COLON, SPACE, STMT),
                    // Action
                    production_rule!(ACTION),
                    // Multiple statements
                    production_rule!(STMT, SEMICOLON, STMT),
                    // If
                    production_rule!(IF, LPAREN, CONDITIONAL, RPAREN, COLON, SPACE, STMT),
                    // IfElse
                    production_rule!(
                        IFELSE,
                        LPAREN,
                        CONDITIONAL,
                        RPAREN,
                        COLON,
                        SPACE,
                        STMT,
                        ELSE,
                        COLON,
                        SPACE,
                        STMT
                    )
                ),
                context_free_production!(
                    CONDITIONAL,
                    production_rule!(FRONT_CLEAR, LPAREN, RPAREN),
                    production_rule!(LEFT_CLEAR, LPAREN, RPAREN),
                    production_rule!(RIGHT_CLEAR, LPAREN, RPAREN),
                    production_rule!(MARKERS_PRESENT, LPAREN, RPAREN),
                    production_rule!(NO_MARKERS_PRESENT, LPAREN, RPAREN),
                    production_rule!(NOT, SPACE, CONDITIONAL)
                ),
                context_free_production!(
                    ACTION,
                    production_rule!(MOVE, LPAREN, RPAREN),
                    production_rule!(TURN_RIGHT, LPAREN, RPAREN),
                    production_rule!(TURN_LEFT, LPAREN, RPAREN),
                    production_rule!(PICK_MARKER, LPAREN, RPAREN),
                    production_rule!(PUT_MARKER, LPAREN, RPAREN)
                ),
                context_free_production!(
                    COUNT,
                    production_rule!(T_1),
                    production_rule!(T_2),
                    production_rule!(T_3),
                    production_rule!(T_4),
                    production_rule!(T_5),
                    production_rule!(T_6),
                    production_rule!(T_7),
                    production_rule!(T_8),
                    production_rule!(T_9),
                    production_rule!(T_1, T_0),
                    production_rule!(T_1, T_1),
                    production_rule!(T_1, T_2),
                    production_rule!(T_1, T_3),
                    production_rule!(T_1, T_4),
                    production_rule!(T_1, T_5),
                    production_rule!(T_1, T_6),
                    production_rule!(T_1, T_7),
                    production_rule!(T_1, T_8),
                    production_rule!(T_1, T_9)
                ),
            ],
        ))
    }
}
