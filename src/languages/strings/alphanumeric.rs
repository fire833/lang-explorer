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
    grammar::{production_rule, Production, ProductionLHS, ProductionRule},
    languages::strings::{terminal_str, GrammarElement, StringValue},
};

terminal_str!(NT_LO_A, "a");
terminal_str!(NT_LO_B, "b");
terminal_str!(NT_LO_C, "c");
terminal_str!(NT_LO_D, "d");
terminal_str!(NT_LO_E, "e");
terminal_str!(NT_LO_F, "f");
terminal_str!(NT_LO_G, "g");
terminal_str!(NT_LO_H, "h");
terminal_str!(NT_LO_I, "i");
terminal_str!(NT_LO_J, "j");
terminal_str!(NT_LO_K, "k");
terminal_str!(NT_LO_L, "l");
terminal_str!(NT_LO_M, "m");
terminal_str!(NT_LO_N, "n");
terminal_str!(NT_LO_O, "o");
terminal_str!(NT_LO_P, "p");
terminal_str!(NT_LO_Q, "q");
terminal_str!(NT_LO_R, "r");
terminal_str!(NT_LO_S, "s");
terminal_str!(NT_LO_T, "t");
terminal_str!(NT_LO_U, "u");
terminal_str!(NT_LO_V, "v");
terminal_str!(NT_LO_W, "w");
terminal_str!(NT_LO_X, "x");
terminal_str!(NT_LO_Y, "y");
terminal_str!(NT_LO_Z, "z");
terminal_str!(NT_UP_A, "A");
terminal_str!(NT_UP_B, "B");
terminal_str!(NT_UP_C, "C");
terminal_str!(NT_UP_D, "D");
terminal_str!(NT_UP_E, "E");
terminal_str!(NT_UP_F, "F");
terminal_str!(NT_UP_G, "G");
terminal_str!(NT_UP_H, "H");
terminal_str!(NT_UP_I, "I");
terminal_str!(NT_UP_J, "J");
terminal_str!(NT_UP_K, "K");
terminal_str!(NT_UP_L, "L");
terminal_str!(NT_UP_M, "M");
terminal_str!(NT_UP_N, "N");
terminal_str!(NT_UP_O, "O");
terminal_str!(NT_UP_P, "P");
terminal_str!(NT_UP_Q, "Q");
terminal_str!(NT_UP_R, "R");
terminal_str!(NT_UP_S, "S");
terminal_str!(NT_UP_T, "T");
terminal_str!(NT_UP_U, "U");
terminal_str!(NT_UP_V, "V");
terminal_str!(NT_UP_W, "W");
terminal_str!(NT_UP_X, "X");
terminal_str!(NT_UP_Y, "Y");
terminal_str!(NT_UP_Z, "Z");
terminal_str!(NT_0, "0");
terminal_str!(NT_1, "1");
terminal_str!(NT_2, "2");
terminal_str!(NT_3, "3");
terminal_str!(NT_4, "4");
terminal_str!(NT_5, "5");
terminal_str!(NT_6, "6");
terminal_str!(NT_7, "7");
terminal_str!(NT_8, "8");
terminal_str!(NT_9, "9");

fn productions_alpha_lower() -> Vec<ProductionRule<StringValue, StringValue>> {
    return vec![
        production_rule!(NT_LO_A),
        production_rule!(NT_LO_B),
        production_rule!(NT_LO_C),
        production_rule!(NT_LO_D),
        production_rule!(NT_LO_E),
        production_rule!(NT_LO_F),
        production_rule!(NT_LO_G),
        production_rule!(NT_LO_H),
        production_rule!(NT_LO_I),
        production_rule!(NT_LO_J),
        production_rule!(NT_LO_K),
        production_rule!(NT_LO_L),
        production_rule!(NT_LO_M),
        production_rule!(NT_LO_N),
        production_rule!(NT_LO_O),
        production_rule!(NT_LO_P),
        production_rule!(NT_LO_Q),
        production_rule!(NT_LO_R),
        production_rule!(NT_LO_S),
        production_rule!(NT_LO_T),
        production_rule!(NT_LO_U),
        production_rule!(NT_LO_V),
        production_rule!(NT_LO_W),
        production_rule!(NT_LO_X),
        production_rule!(NT_LO_Y),
        production_rule!(NT_LO_Z),
    ];
}

fn productions_alpha_upper() -> Vec<ProductionRule<StringValue, StringValue>> {
    return vec![
        production_rule!(NT_UP_A),
        production_rule!(NT_UP_B),
        production_rule!(NT_UP_C),
        production_rule!(NT_UP_D),
        production_rule!(NT_UP_E),
        production_rule!(NT_UP_F),
        production_rule!(NT_UP_G),
        production_rule!(NT_UP_H),
        production_rule!(NT_UP_I),
        production_rule!(NT_UP_J),
        production_rule!(NT_UP_K),
        production_rule!(NT_UP_L),
        production_rule!(NT_UP_M),
        production_rule!(NT_UP_N),
        production_rule!(NT_UP_O),
        production_rule!(NT_UP_P),
        production_rule!(NT_UP_Q),
        production_rule!(NT_UP_R),
        production_rule!(NT_UP_S),
        production_rule!(NT_UP_T),
        production_rule!(NT_UP_U),
        production_rule!(NT_UP_V),
        production_rule!(NT_UP_W),
        production_rule!(NT_UP_X),
        production_rule!(NT_UP_Y),
        production_rule!(NT_UP_Z),
    ];
}

fn productions_digits() -> Vec<ProductionRule<StringValue, StringValue>> {
    return vec![
        production_rule!(NT_0),
        production_rule!(NT_1),
        production_rule!(NT_2),
        production_rule!(NT_3),
        production_rule!(NT_4),
        production_rule!(NT_5),
        production_rule!(NT_6),
        production_rule!(NT_7),
        production_rule!(NT_8),
        production_rule!(NT_9),
    ];
}

pub fn alphanumeric_character_production_context_free(
    root: StringValue,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![];

    productions.append(&mut productions_alpha_lower());
    productions.append(&mut productions_alpha_upper());
    productions.append(&mut productions_digits());

    return Production::new(ProductionLHS::new_context_free(root), productions);
}

pub fn alpha_lower_character_production_context_free(
    root: StringValue,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![];

    productions.append(&mut productions_alpha_lower());

    return Production::new(ProductionLHS::new_context_free(root), productions);
}

pub fn alpha_upper_character_production_context_free(
    root: StringValue,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![];

    productions.append(&mut productions_alpha_upper());

    return Production::new(ProductionLHS::new_context_free(root), productions);
}

pub fn numeric_character_production_context_free(
    root: StringValue,
) -> Production<StringValue, StringValue> {
    let mut productions = vec![];

    productions.append(&mut productions_digits());

    return Production::new(ProductionLHS::new_context_free(root), productions);
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

    return Production::new(ProductionLHS::new_context_free(root), productions);
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

    return Production::new(ProductionLHS::new_context_free(root), productions);
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

    return Production::new(ProductionLHS::new_context_free(root), productions);
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

    return Production::new(ProductionLHS::new_context_free(root), productions);
}
