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
            alphanumeric::{T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9},
            nterminal_str, terminal_str, StringValue, COLON, LPAREN, RPAREN, SEMICOLON, SPACE,
        },
        GrammarBuilder, NOPGrammarState,
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
terminal_str!(FRONT_CLEAR, "frontIsClear()");
terminal_str!(LEFT_CLEAR, "leftIsClear()");
terminal_str!(RIGHT_CLEAR, "rightIsClear()");
terminal_str!(MARKERS_PRESENT, "markersPresent()");
terminal_str!(NO_MARKERS_PRESENT, "noMarkersPresent()");
terminal_str!(NOT, "not");
terminal_str!(MOVE, "move()");
terminal_str!(TURN_RIGHT, "turnRight()");
terminal_str!(TURN_LEFT, "turnLeft()");
terminal_str!(PICK_MARKER, "pickMarker()");
terminal_str!(PUT_MARKER, "putMarker()");

// Numbers
terminal_str!(T_10, "10");
terminal_str!(T_11, "11");
terminal_str!(T_12, "12");
terminal_str!(T_13, "13");
terminal_str!(T_14, "14");
terminal_str!(T_15, "15");
terminal_str!(T_16, "16");
terminal_str!(T_17, "17");
terminal_str!(T_18, "18");
terminal_str!(T_19, "19");

pub struct KarelLanguage;

/// Parameters for Karel Language.
#[derive(Default, Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct KarelLanguageParameters {}

impl GrammarBuilder for KarelLanguage {
    type Term = StringValue;
    type NTerm = StringValue;
    type Params<'de> = KarelLanguageParameters;
    type State = NOPGrammarState;

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
                    production_rule!(5, WHILE, LPAREN, CONDITIONAL, RPAREN, COLON, SPACE, STMT),
                    // Repeat loop
                    production_rule!(5, REPEAT, LPAREN, COUNT, RPAREN, COLON, SPACE, STMT),
                    // Action
                    production_rule!(9, ACTION),
                    // Multiple statements
                    production_rule!(3, STMT, SEMICOLON, STMT),
                    // If
                    production_rule!(5, IF, LPAREN, CONDITIONAL, RPAREN, COLON, SPACE, STMT),
                    // IfElse
                    production_rule!(
                        5,
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
                    production_rule!(5, FRONT_CLEAR),
                    production_rule!(5, LEFT_CLEAR),
                    production_rule!(5, RIGHT_CLEAR),
                    production_rule!(5, MARKERS_PRESENT),
                    production_rule!(5, NO_MARKERS_PRESENT),
                    production_rule!(2, NOT, SPACE, CONDITIONAL)
                ),
                context_free_production!(
                    ACTION,
                    production_rule!(MOVE),
                    production_rule!(TURN_RIGHT),
                    production_rule!(TURN_LEFT),
                    production_rule!(PICK_MARKER),
                    production_rule!(PUT_MARKER)
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
                    production_rule!(T_10),
                    production_rule!(T_11),
                    production_rule!(T_12),
                    production_rule!(T_13),
                    production_rule!(T_14),
                    production_rule!(T_15),
                    production_rule!(T_16),
                    production_rule!(T_17),
                    production_rule!(T_18),
                    production_rule!(T_19)
                ),
            ],
            "karel".into(),
        ))
    }
}
