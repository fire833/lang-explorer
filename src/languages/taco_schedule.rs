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

use crate::grammar::{Grammar, GrammarElement, Production, ProductionLHS, ProductionRule};

use super::StringValue;

const POS_OP: &str = "pos";
const FUSE_OP: &str = "fuse";
const SPLIT_OP: &str = "split";
const REORDER_OP: &str = "reorder";

pub fn taco_schedule_grammar() -> Grammar<StringValue, StringValue> {
    Grammar::new(
        "Entrypoint".into(),
        vec![
            Production::new(
                ProductionLHS::new_context_free("entrypoint".into()),
                vec![
                    ProductionRule::new(vec![GrammarElement::Epsilon]),
                    ProductionRule::new(vec![
                        GrammarElement::Terminal(StringValue::from("pos")),
                        GrammarElement::NonTerminal(StringValue::from("A")),
                    ]),
                ],
            ),
            Production::new(
                ProductionLHS::new_context_free("A".into()),
                vec![ProductionRule::new(vec![GrammarElement::Epsilon])],
            ),
        ],
    )
}
