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

use crate::{
    grammar::{
        elem::GrammarElement, lhs::ProductionLHS, prod::production_rule, prod::Production,
        rule::ProductionRule,
    },
    languages::strings::{terminal_str, StringValue},
};

terminal_str!(pub, T_LO_A, "a");
terminal_str!(pub, T_LO_B, "b");
terminal_str!(pub, T_LO_C, "c");
terminal_str!(pub, T_LO_D, "d");
terminal_str!(pub, T_LO_E, "e");
terminal_str!(pub, T_LO_F, "f");
terminal_str!(pub, T_LO_G, "g");
terminal_str!(pub, T_LO_H, "h");
terminal_str!(pub, T_LO_I, "i");
terminal_str!(pub, T_LO_J, "j");
terminal_str!(pub, T_LO_K, "k");
terminal_str!(pub, T_LO_L, "l");
terminal_str!(pub, T_LO_M, "m");
terminal_str!(pub, T_LO_N, "n");
terminal_str!(pub, T_LO_O, "o");
terminal_str!(pub, T_LO_P, "p");
terminal_str!(pub, T_LO_Q, "q");
terminal_str!(pub, T_LO_R, "r");
terminal_str!(pub, T_LO_S, "s");
terminal_str!(pub, T_LO_T, "t");
terminal_str!(pub, T_LO_U, "u");
terminal_str!(pub, T_LO_V, "v");
terminal_str!(pub, T_LO_W, "w");
terminal_str!(pub, T_LO_X, "x");
terminal_str!(pub, T_LO_Y, "y");
terminal_str!(pub, T_LO_Z, "z");
terminal_str!(pub, T_UP_A, "A");
terminal_str!(pub, T_UP_B, "B");
terminal_str!(pub, T_UP_C, "C");
terminal_str!(pub, T_UP_D, "D");
terminal_str!(pub, T_UP_E, "E");
terminal_str!(pub, T_UP_F, "F");
terminal_str!(pub, T_UP_G, "G");
terminal_str!(pub, T_UP_H, "H");
terminal_str!(pub, T_UP_I, "I");
terminal_str!(pub, T_UP_J, "J");
terminal_str!(pub, T_UP_K, "K");
terminal_str!(pub, T_UP_L, "L");
terminal_str!(pub, T_UP_M, "M");
terminal_str!(pub, T_UP_N, "N");
terminal_str!(pub, T_UP_O, "O");
terminal_str!(pub, T_UP_P, "P");
terminal_str!(pub, T_UP_Q, "Q");
terminal_str!(pub, T_UP_R, "R");
terminal_str!(pub, T_UP_S, "S");
terminal_str!(pub, T_UP_T, "T");
terminal_str!(pub, T_UP_U, "U");
terminal_str!(pub, T_UP_V, "V");
terminal_str!(pub, T_UP_W, "W");
terminal_str!(pub, T_UP_X, "X");
terminal_str!(pub, T_UP_Y, "Y");
terminal_str!(pub, T_UP_Z, "Z");
terminal_str!(pub, T_0, "0");
terminal_str!(pub, T_1, "1");
terminal_str!(pub, T_2, "2");
terminal_str!(pub, T_3, "3");
terminal_str!(pub, T_4, "4");
terminal_str!(pub, T_5, "5");
terminal_str!(pub, T_6, "6");
terminal_str!(pub, T_7, "7");
terminal_str!(pub, T_8, "8");
terminal_str!(pub, T_9, "9");

fn productions_alpha_lower() -> Vec<ProductionRule<StringValue, StringValue>> {
    vec![
        production_rule!(T_LO_A),
        production_rule!(T_LO_B),
        production_rule!(T_LO_C),
        production_rule!(T_LO_D),
        production_rule!(T_LO_E),
        production_rule!(T_LO_F),
        production_rule!(T_LO_G),
        production_rule!(T_LO_H),
        production_rule!(T_LO_I),
        production_rule!(T_LO_J),
        production_rule!(T_LO_K),
        production_rule!(T_LO_L),
        production_rule!(T_LO_M),
        production_rule!(T_LO_N),
        production_rule!(T_LO_O),
        production_rule!(T_LO_P),
        production_rule!(T_LO_Q),
        production_rule!(T_LO_R),
        production_rule!(T_LO_S),
        production_rule!(T_LO_T),
        production_rule!(T_LO_U),
        production_rule!(T_LO_V),
        production_rule!(T_LO_W),
        production_rule!(T_LO_X),
        production_rule!(T_LO_Y),
        production_rule!(T_LO_Z),
    ]
}

fn productions_alpha_upper() -> Vec<ProductionRule<StringValue, StringValue>> {
    vec![
        production_rule!(T_UP_A),
        production_rule!(T_UP_B),
        production_rule!(T_UP_C),
        production_rule!(T_UP_D),
        production_rule!(T_UP_E),
        production_rule!(T_UP_F),
        production_rule!(T_UP_G),
        production_rule!(T_UP_H),
        production_rule!(T_UP_I),
        production_rule!(T_UP_J),
        production_rule!(T_UP_K),
        production_rule!(T_UP_L),
        production_rule!(T_UP_M),
        production_rule!(T_UP_N),
        production_rule!(T_UP_O),
        production_rule!(T_UP_P),
        production_rule!(T_UP_Q),
        production_rule!(T_UP_R),
        production_rule!(T_UP_S),
        production_rule!(T_UP_T),
        production_rule!(T_UP_U),
        production_rule!(T_UP_V),
        production_rule!(T_UP_W),
        production_rule!(T_UP_X),
        production_rule!(T_UP_Y),
        production_rule!(T_UP_Z),
    ]
}

fn productions_digits() -> Vec<ProductionRule<StringValue, StringValue>> {
    vec![
        production_rule!(T_0),
        production_rule!(T_1),
        production_rule!(T_2),
        production_rule!(T_3),
        production_rule!(T_4),
        production_rule!(T_5),
        production_rule!(T_6),
        production_rule!(T_7),
        production_rule!(T_8),
        production_rule!(T_9),
    ]
}

pub fn alphanumeric_character_production_context_free(
    root: StringValue,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![];

    productions.append(&mut productions_alpha_lower());
    productions.append(&mut productions_alpha_upper());
    productions.append(&mut productions_digits());

    Production::new(ProductionLHS::new_context_free(root), productions)
}

pub fn alpha_character_production_context_free(
    root: StringValue,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![];

    productions.append(&mut productions_alpha_lower());
    productions.append(&mut productions_alpha_upper());

    Production::new(ProductionLHS::new_context_free(root), productions)
}

pub fn alpha_lower_character_production_context_free(
    root: StringValue,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![];

    productions.append(&mut productions_alpha_lower());

    Production::new(ProductionLHS::new_context_free(root), productions)
}

pub fn alpha_upper_character_production_context_free(
    root: StringValue,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![];

    productions.append(&mut productions_alpha_upper());

    Production::new(ProductionLHS::new_context_free(root), productions)
}

pub fn numeric_character_production_context_free(
    root: StringValue,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![];

    productions.append(&mut productions_digits());

    Production::new(ProductionLHS::new_context_free(root), productions)
}

pub fn alphanumeric_string_production_context_free(
    root: StringValue,
    iterator: GrammarElement<StringValue, StringValue>,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![
        production_rule!(iterator.clone(), iterator.clone()),
        production_rule!(iterator.clone()),
    ];

    productions.append(&mut productions_alpha_lower());
    productions.append(&mut productions_alpha_upper());
    productions.append(&mut productions_digits());

    Production::new(ProductionLHS::new_context_free(root), productions)
}

pub fn alpha_string_production_context_free(
    root: StringValue,
    iterator: GrammarElement<StringValue, StringValue>,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![
        production_rule!(iterator.clone(), iterator.clone()),
        production_rule!(iterator.clone()),
    ];

    productions.append(&mut productions_alpha_lower());
    productions.append(&mut productions_alpha_upper());

    Production::new(ProductionLHS::new_context_free(root), productions)
}

pub fn lowercase_string_production_context_free(
    root: StringValue,
    iterator: GrammarElement<StringValue, StringValue>,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![
        production_rule!(iterator.clone(), iterator.clone()),
        production_rule!(iterator.clone()),
    ];

    productions.append(&mut productions_alpha_lower());

    Production::new(ProductionLHS::new_context_free(root), productions)
}

pub fn uppercase_string_production_context_free(
    root: StringValue,
    iterator: GrammarElement<StringValue, StringValue>,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![
        production_rule!(iterator.clone(), iterator.clone()),
        production_rule!(iterator.clone()),
    ];

    productions.append(&mut productions_alpha_upper());

    Production::new(ProductionLHS::new_context_free(root), productions)
}
