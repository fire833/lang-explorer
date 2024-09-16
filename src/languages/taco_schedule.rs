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

use crate::grammar::{
    production_rule, Grammar, GrammarElement, Production, ProductionLHS, ProductionRule,
};

use super::strings::{nterminal_str, terminal_str, StringValue, COMMA, EPSILON, LPAREN, RPAREN};

// Non terminals for this grammar.
nterminal_str!(NT_ENTRYPOINT, "entrypoint");
nterminal_str!(NT_RULE, "rule");
nterminal_str!(NT_ASSEMBLE_RULE, "assemble_rule");
nterminal_str!(NT_INDEX_VARIABLE, "index_variable");

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

// Terminal arguments for this grammar
terminal_str!(ASSEMBLE_APPEND, "Append");
terminal_str!(ASSEMBLE_INSERT, "Insert");

pub struct TacoGrammar {
    index_variables: Vec<String>,
}

impl TacoGrammar {
    pub fn new(index_variables: Vec<String>) -> Self {
        Self { index_variables }
    }

    pub fn taco_schedule_grammar(&self) -> Grammar<StringValue, StringValue> {
        let index_productions: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        // for var in self.index_variables.iter() {
        //     // Store this variable in the heap.
        //     let b = Box::new(var.clone().as_str());
        //     let term = GrammarElement::Terminal(StringValue::from(b));
        //     index_productions.push(production_rule!(term));
        // }

        Grammar::new(
            "entrypoint".into(),
            vec![
                // Entrypoint rule
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_ENTRYPOINT),
                    vec![
                        // Optionally create no rules
                        production_rule!(EPSILON),
                        // Or one rule
                        production_rule!(NT_RULE),
                        // Or many rules
                        production_rule!(NT_ENTRYPOINT, COMMA, NT_RULE),
                    ],
                ),
                // Rule definition rule
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_RULE),
                    vec![
                        // pos(index_variable, derived_index_var, tensor)
                        production_rule!(POS_OP, LPAREN, NT_INDEX_VARIABLE, COMMA, RPAREN),
                        // fuse(index_variable, index_variable, fused_index_variable)
                        production_rule!(
                            FUSE_OP,
                            LPAREN,
                            NT_INDEX_VARIABLE,
                            COMMA,
                            NT_INDEX_VARIABLE,
                            COMMA,
                            RPAREN
                        ),
                        // split()
                        production_rule!(SPLIT_OP, LPAREN, NT_INDEX_VARIABLE, COMMA, RPAREN), // Incomplete
                        // reorder(index_variable)
                        production_rule!(REORDER_OP, LPAREN, NT_INDEX_VARIABLE, RPAREN),
                        // divide()
                        production_rule!(DIVIDE_OP, LPAREN, RPAREN),
                        // precompute()
                        production_rule!(PRECOMPUTE_OP, LPAREN, RPAREN),
                        // unroll()
                        production_rule!(UNROLL_OP, LPAREN, RPAREN),
                        // bound()
                        production_rule!(BOUND_OP, LPAREN, RPAREN),
                        // parallelize()
                        production_rule!(PARALLELIZE_OP, LPAREN, RPAREN),
                        // assemble()
                        production_rule!(ASSEMBLE_OP, LPAREN, COMMA, NT_ASSEMBLE_RULE, RPAREN),
                    ],
                ),
                // Assemble Operation Rule.
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_ASSEMBLE_RULE),
                    vec![
                        production_rule!(ASSEMBLE_APPEND),
                        production_rule!(ASSEMBLE_INSERT),
                    ],
                ),
                // Index Variable Rule,
                Production::new(
                    ProductionLHS::new_context_free_elem(NT_INDEX_VARIABLE),
                    index_productions,
                ),
            ],
        )
    }
}

// impl<T, I> GrammarBuilder<T, I> for TacoGrammar
// where
//     T: Terminal,
//     I: NonTerminal,
// {
//     fn generate_grammar(&self) -> Result<Grammar<T, I>, LangExplorerError> {
//         Ok(self.taco_schedule_grammar())
//     }
// }
