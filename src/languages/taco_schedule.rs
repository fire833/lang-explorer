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

use super::{nterminal_str, terminal_str, StringValue};

const EPSILON: GrammarElement<StringValue, StringValue> = GrammarElement::Epsilon;

// Non terminals for this grammar.
nterminal_str!(NT_ENTRYPOINT, "entrypoint");
nterminal_str!(NT_RULE, "rule");

// Terminals for this grammar.
terminal_str!(POS_OP, "pos");
terminal_str!(FUSE_OP, "fuse");
terminal_str!(SPLIT_OP, "split");
terminal_str!(REORDER_OP, "reorder");
terminal_str!(DIVIDE_OP, "divide");
terminal_str!(PRECOMPUTE_OP, "precompute");
terminal_str!(UNROLL_OP, "unroll");
terminal_str!(BOUND_OP, "bound");
terminal_str!(PARALLELIZE_OP, "parallelize");
terminal_str!(ASSEMBLE_OP, "assemble");

terminal_str!(COMMA, ",");

pub fn taco_schedule_grammar() -> Grammar<StringValue, StringValue> {
    Grammar::new(
        "Entrypoint".into(),
        vec![
            Production::new(
                ProductionLHS::new_context_free("entrypoint".into()),
                vec![
                    // Optionally create no rules
                    ProductionRule::new(vec![EPSILON]),
                    // Or one rule
                    ProductionRule::new(vec![NT_RULE]),
                    // Or many rules
                    ProductionRule::new(vec![NT_ENTRYPOINT, COMMA, NT_RULE]),
                ],
            ),
            Production::new(
                ProductionLHS::new_context_free("rule".into()),
                vec![ProductionRule::new(vec![GrammarElement::Epsilon])],
            ),
        ],
    )
}
