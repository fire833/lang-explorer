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

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    errors::LangExplorerError,
    evaluators::{Evaluator, Metric},
    grammar::{
        elem::GrammarElement,
        grammar::Grammar,
        lhs::ProductionLHS,
        prod::{context_free_production, production_rule, Production},
        rule::ProductionRule,
        NonTerminal, Terminal,
    },
    languages::{GrammarState, NOPGrammarState},
};

use super::{
    strings::{nterminal_str, terminal_str, StringValue, COMMA, EPSILON, LPAREN, RPAREN},
    GrammarBuilder,
};

// Non terminals for this grammar.
nterminal_str!(NT_ENTRYPOINT, "entrypoint");
nterminal_str!(NT_EXPANSION, "expansion");
nterminal_str!(NT_RULE, "rule");
nterminal_str!(NT_ASSEMBLE_STRATEGY, "assemble_strategy");
nterminal_str!(NT_PARALLELIZE_HW, "parallelize_hw");
nterminal_str!(NT_PARALLELIZE_RACES, "parallelize_races");
nterminal_str!(NT_INDEX_VARIABLE, "index_variable");
nterminal_str!(NT_WORKSPACE_INDEX_VARIABLE, "workspace_index_variable");
nterminal_str!(NT_FUSED_INDEX_VARIABLE, "fused_index_variable");
nterminal_str!(NT_UNROLL_FACTOR, "unroll_factor");
nterminal_str!(NT_DIVIDE_FACTOR, "divide_factor");
nterminal_str!(NT_SPLIT_FACTOR, "split_factor");

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

// Terminal Assemble operation arguments for this grammar
terminal_str!(ASSEMBLE_APPEND, "Append");
terminal_str!(ASSEMBLE_INSERT, "Insert");

// Terminal Parallelize Hardware operation arguments for this grammar
terminal_str!(PARALLELIZE_HW_NOTPARALLEL, "NotParallel");
terminal_str!(PARALLELIZE_HW_CPUTHREAD, "CPUThread");
terminal_str!(PARALLELIZE_HW_CPUVECTOR, "CPUVector");
terminal_str!(PARALLELIZE_HW_GPUWARP, "GPUWarp");

// Terminal Parallelize Race Strategy operation arguments for this grammar.
terminal_str!(PARALLELIZE_RACE_IGNORE, "IgnoreRaces");
terminal_str!(PARALLELIZE_RACE_NORACES, "NoRaces");
terminal_str!(PARALLELIZE_RACE_ATOMICS, "Atomics");
terminal_str!(PARALLELIZE_RACE_TEMP, "Temporary");
terminal_str!(PARALLELIZE_RACE_PREDUCE, "ParallelReduction");

pub struct TacoScheduleLanguage;

pub struct TacoScheduleState {}

impl Default for TacoScheduleState {
    fn default() -> Self {
        TacoScheduleState {}
    }
}

impl<T: Terminal, I: NonTerminal> GrammarState<T, I> for TacoScheduleState {
    fn apply_context<'a>(&mut self, prod: &'a Production<T, I>) -> Option<Production<T, I>> {
        None
    }

    fn update(&mut self, rule: &ProductionRule<T, I>) {
        todo!()
    }
}

/// Parameters for Taco Schedule Language.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TacoScheduleLanguageParams {
    #[serde(alias = "version")]
    version: TacoScheduleLanguageVersion,

    index_variables: Vec<String>,
    workspace_index_variables: Vec<String>,
    fused_index_variables: Vec<String>,
    split_factor_variables: Vec<String>,
    divide_factor_variables: Vec<String>,
    unroll_factor_variables: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum TacoScheduleLanguageVersion {
    /// Generate schedules with the context free version of this grammar.
    /// Please note, this version can lead to syntactically incorrect programs.
    ContextFreeV1,
}

impl Default for TacoScheduleLanguageVersion {
    fn default() -> Self {
        Self::ContextFreeV1
    }
}

impl Default for TacoScheduleLanguageParams {
    fn default() -> Self {
        Self {
            version: Default::default(),
            index_variables: vec!["i".into(), "j".into(), "k".into(), "x".into(), "y".into()],
            workspace_index_variables: vec!["l".into(), "m".into(), "n".into()],
            fused_index_variables: vec!["o".into(), "p".into()],
            split_factor_variables: vec![
                "2".into(),
                "4".into(),
                "8".into(),
                "16".into(),
                "32".into(),
            ],
            divide_factor_variables: vec!["t".into(), "u".into()],
            unroll_factor_variables: vec!["v".into(), "w".into(), "z".into()],
        }
    }
}

impl TacoScheduleLanguageParams {
    pub fn new(
        index_variables: Vec<String>,
        workspace_index_variables: Vec<String>,
        fused_index_variables: Vec<String>,
        split_factors: Vec<String>,
        divide_factors: Vec<String>,
        unroll_factors: Vec<String>,
    ) -> Self {
        Self {
            version: Default::default(),
            index_variables,
            workspace_index_variables,
            fused_index_variables,
            split_factor_variables: split_factors,
            divide_factor_variables: divide_factors,
            unroll_factor_variables: unroll_factors,
        }
    }
}

impl GrammarBuilder for TacoScheduleLanguage {
    type Term = StringValue;
    type NTerm = StringValue;
    type Params<'de> = TacoScheduleLanguageParams;
    type State = NOPGrammarState;

    fn generate_grammar<'de>(
        params: Self::Params<'de>,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError> {
        let mut index_productions: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        for var in params.index_variables.iter() {
            // Store this variable in the heap.
            let term = GrammarElement::Terminal(var.into());
            index_productions.push(production_rule!(term));
        }

        let mut workspace_index_productions: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        for var in params.workspace_index_variables.iter() {
            let term = GrammarElement::Terminal(var.into());
            workspace_index_productions.push(production_rule!(term));
        }

        let mut fused_index_productions: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        for var in params.fused_index_variables.iter() {
            let term = GrammarElement::Terminal(var.into());
            fused_index_productions.push(production_rule!(term));
        }

        let mut split_factor_productions: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        for var in params.split_factor_variables.iter() {
            let term = GrammarElement::Terminal(var.into());
            split_factor_productions.push(production_rule!(term));
        }

        let mut divide_factor_productions: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        for var in params.divide_factor_variables.iter() {
            let term = GrammarElement::Terminal(var.into());
            divide_factor_productions.push(production_rule!(term));
        }

        let mut unroll_factor_productions: Vec<ProductionRule<StringValue, StringValue>> = vec![];
        for var in params.unroll_factor_variables.iter() {
            let term = GrammarElement::Terminal(var.into());
            unroll_factor_productions.push(production_rule!(term));
        }

        let grammar = match params.version {
            TacoScheduleLanguageVersion::ContextFreeV1 => Grammar::new(
                "entrypoint".into(),
                vec![
                    // Entrypoint rule
                    context_free_production!(
                        NT_ENTRYPOINT, // Optionally create no rules
                        production_rule!(2, EPSILON),
                        // Or generate rule
                        production_rule!(40, NT_EXPANSION),
                        // Or generate many rules
                        production_rule!(58, NT_RULE, COMMA, NT_EXPANSION)
                    ),
                    context_free_production!(
                        NT_EXPANSION,
                        production_rule!(NT_RULE, COMMA, NT_EXPANSION),
                        production_rule!(NT_RULE)
                    ),
                    // Rule definition rule
                    context_free_production!(
                        NT_RULE,
                        // pos(index_variable, derived_index_var, tensor)
                        // production_rule!(POS_OP, LPAREN, NT_INDEX_VARIABLE, COMMA, RPAREN),
                        // fuse(index_variable, index_variable, fused_index_variable)
                        production_rule!(
                            FUSE_OP,
                            LPAREN,
                            NT_INDEX_VARIABLE,
                            COMMA,
                            NT_INDEX_VARIABLE,
                            COMMA,
                            NT_FUSED_INDEX_VARIABLE,
                            RPAREN
                        ),
                        // split(index_variable, outer_index_variable, inner_index_variable, split_factor)
                        production_rule!(
                            SPLIT_OP,
                            LPAREN,
                            NT_INDEX_VARIABLE,
                            COMMA,
                            NT_INDEX_VARIABLE,
                            COMMA,
                            NT_SPLIT_FACTOR,
                            RPAREN
                        ),
                        // reorder(index_variable)
                        production_rule!(REORDER_OP, LPAREN, NT_INDEX_VARIABLE, RPAREN),
                        // divide(index_variable, outer_index_variable, inner_index_variable, divide_factor)
                        production_rule!(
                            DIVIDE_OP,
                            LPAREN,
                            NT_INDEX_VARIABLE,
                            COMMA,
                            NT_INDEX_VARIABLE,
                            COMMA,
                            NT_DIVIDE_FACTOR,
                            RPAREN
                        ),
                        // precompute(expr, index_variable, workspace_index_variable)
                        production_rule!(
                            PRECOMPUTE_OP,
                            LPAREN,
                            NT_INDEX_VARIABLE,
                            COMMA,
                            NT_WORKSPACE_INDEX_VARIABLE,
                            RPAREN
                        ),
                        // unroll(index_variable, unroll_factor)
                        production_rule!(
                            UNROLL_OP,
                            LPAREN,
                            NT_INDEX_VARIABLE,
                            COMMA,
                            NT_UNROLL_FACTOR,
                            RPAREN
                        ),
                        // bound() // appears to be not yet supported in taco.
                        // production_rule!(BOUND_OP, LPAREN, RPAREN),
                        // parallelize(index_variable, hardware, race_strategy)
                        production_rule!(
                            PARALLELIZE_OP,
                            LPAREN,
                            NT_INDEX_VARIABLE,
                            COMMA,
                            NT_PARALLELIZE_HW,
                            COMMA,
                            NT_PARALLELIZE_RACES,
                            RPAREN
                        ) // assemble()
                          // production_rule!(ASSEMBLE_OP, LPAREN, COMMA, NT_ASSEMBLE_STRATEGY, RPAREN)
                    ),
                    // Assemble Operation Rule.
                    Production::new(
                        ProductionLHS::new_context_free_elem(NT_ASSEMBLE_STRATEGY),
                        vec![
                            production_rule!(ASSEMBLE_APPEND),
                            production_rule!(ASSEMBLE_INSERT),
                        ],
                    ),
                    // Parallelize Hardware Rule.
                    Production::new(
                        ProductionLHS::new_context_free_elem(NT_PARALLELIZE_HW),
                        vec![
                            production_rule!(PARALLELIZE_HW_CPUTHREAD),
                            production_rule!(PARALLELIZE_HW_CPUVECTOR),
                            production_rule!(PARALLELIZE_HW_GPUWARP),
                            production_rule!(PARALLELIZE_HW_NOTPARALLEL),
                        ],
                    ),
                    // Parallelize Race Conditions Rule.
                    Production::new(
                        ProductionLHS::new_context_free_elem(NT_PARALLELIZE_RACES),
                        vec![
                            production_rule!(PARALLELIZE_RACE_IGNORE),
                            production_rule!(PARALLELIZE_RACE_TEMP),
                            production_rule!(PARALLELIZE_RACE_NORACES),
                            production_rule!(PARALLELIZE_RACE_PREDUCE),
                            production_rule!(PARALLELIZE_RACE_ATOMICS),
                        ],
                    ),
                    // Index variable rule
                    Production::new(
                        ProductionLHS::new_context_free_elem(NT_INDEX_VARIABLE),
                        index_productions,
                    ),
                    // Workspace index variable rule
                    Production::new(
                        ProductionLHS::new_context_free_elem(NT_WORKSPACE_INDEX_VARIABLE),
                        workspace_index_productions,
                    ),
                    // Fused index variable
                    Production::new(
                        ProductionLHS::new_context_free_elem(NT_FUSED_INDEX_VARIABLE),
                        fused_index_productions,
                    ),
                    // Unroll factor rule
                    Production::new(
                        ProductionLHS::new_context_free_elem(NT_UNROLL_FACTOR),
                        unroll_factor_productions,
                    ),
                    // Split factor rule
                    Production::new(
                        ProductionLHS::new_context_free_elem(NT_SPLIT_FACTOR),
                        split_factor_productions,
                    ),
                    // Divide factor rule
                    Production::new(
                        ProductionLHS::new_context_free_elem(NT_DIVIDE_FACTOR),
                        divide_factor_productions,
                    ),
                ],
                "tacosched".into(),
            ),
        };

        Ok(grammar)
    }
}

impl Metric for u64 {}

#[async_trait]
impl Evaluator for TacoScheduleLanguage {
    type Metric = u64;

    async fn evaluate(&self, _program: Vec<u8>) -> Result<Self::Metric, LangExplorerError> {
        Err(LangExplorerError::General("unimplemented".into()))
    }
}
