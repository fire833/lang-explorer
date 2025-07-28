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

use std::{
    fmt::Debug,
    hash::{DefaultHasher, Hash, Hasher},
    vec,
};

use crate::grammar::{elem::GrammarElement, program::ProgramInstance, NonTerminal, Terminal};

#[allow(unused)]
use crate::languages::strings::{nterminal_str, terminal_str, StringValue};

/// A wrapper type for left-hand sides of grammars, which can include grammars that are
/// context-sensitive. This type allows you to provide optional prefix and suffix
/// grammar elements around the non-terminal as context for the expander.
#[derive(Clone, PartialEq, Eq)]
pub struct ProductionLHS<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    /// optional prefix context for the rule.
    pub prefix: Vec<GrammarElement<T, I>>,

    /// non-terminal for the rule.
    pub non_terminal: I,

    /// optional suffix context for the rule.
    pub suffix: Vec<GrammarElement<T, I>>,

    /// the entire token list for this LHS, compute it once
    /// on init since this will have to be used quite often.
    /// and its cheaper than computing it on the fly.
    /// And make it optional as a cheap way of checking whether
    /// this LHS is context free. If empty, then this LHS can be
    /// considered to be context-free.
    full_token_list: Vec<GrammarElement<T, I>>,
}

impl<T, I> ProductionLHS<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    pub fn new_context_free_elem(non_terminal: GrammarElement<T, I>) -> Self {
        if let GrammarElement::NonTerminal(nt) = non_terminal {
            Self::new_context_free(nt)
        } else {
            panic!("grammar element must be a non-terminal");
        }
    }

    /// Create a new ProductionLHS with no context, only provide a non-terminal
    /// for expansion.
    pub fn new_context_free(non_terminal: I) -> Self {
        Self {
            prefix: vec![],
            non_terminal: non_terminal.clone(),
            suffix: vec![],
            full_token_list: vec![GrammarElement::NonTerminal(non_terminal)],
        }
    }

    pub fn new_with_prefix_single(prefix: GrammarElement<T, I>, non_terminal: I) -> Self {
        Self::new_with_prefix_list(vec![prefix], non_terminal)
    }

    /// Create a new ProductionLHS with prefix context.
    pub fn new_with_prefix_list(prefix: Vec<GrammarElement<T, I>>, non_terminal: I) -> Self {
        let mut tokens = prefix.clone();
        tokens.push(GrammarElement::NonTerminal(non_terminal.clone()));

        Self {
            prefix,
            non_terminal,
            suffix: vec![],
            full_token_list: tokens,
        }
    }

    pub fn new_with_suffix_single(suffix: GrammarElement<T, I>, non_terminal: I) -> Self {
        Self::new_with_suffix_list(vec![suffix], non_terminal)
    }

    /// Create a new ProductionLHS with suffix context.
    pub fn new_with_suffix_list(suffix: Vec<GrammarElement<T, I>>, non_terminal: I) -> Self {
        let mut tokens = vec![GrammarElement::NonTerminal(non_terminal.clone())];
        tokens.append(&mut suffix.clone());

        Self {
            prefix: vec![],
            non_terminal,
            suffix,
            full_token_list: tokens,
        }
    }

    /// Create a new ProductionLHS with both prefix and suffix context.
    pub fn new_with_prefix_and_suffix(
        prefix: Vec<GrammarElement<T, I>>,
        non_terminal: I,
        suffix: Vec<GrammarElement<T, I>>,
    ) -> Self {
        let mut tokens = prefix.clone();
        tokens.push(GrammarElement::NonTerminal(non_terminal.clone()));
        tokens.append(&mut suffix.clone());

        Self {
            prefix,
            non_terminal,
            suffix,
            full_token_list: tokens,
        }
    }

    pub(super) fn get_all_context_instances(
        &self,
        frontier: &Vec<&mut ProgramInstance<T, I>>,
    ) -> Vec<usize> {
        let mut instances = vec![];
        let tokens = &self.full_token_list;

        for idx in 0..frontier.len() - (tokens.len() - 1) {
            let mut found = true;
            for (offset, item) in tokens.iter().enumerate() {
                if *item != frontier[idx + offset].get_node() {
                    found = false;
                    break;
                }
            }

            if found {
                instances.push(idx + self.prefix.len());
            }
        }

        instances
    }

    /// Check if an LHS contains a particular element.
    pub fn contains(&self, elem: &GrammarElement<T, I>) -> bool {
        let mut contains = false;
        self.full_token_list
            .iter()
            .for_each(|item| contains |= *elem == *item);

        contains
    }

    pub fn is_context_sensitive(&self) -> bool {
        self.prefix.len() > 0 || self.suffix.len() > 0
    }
}

#[test]
fn test_get_all_context_instances() {
    nterminal_str!(FOO, "foo");
    terminal_str!(BAR, "bar");
    terminal_str!(BAR2, "bar2");
    terminal_str!(BAR3, "bar3");
    nterminal_str!(BAZ, "baz");

    macro_rules! pi {
        ($s:expr) => {
            &mut ProgramInstance::new($s, 1)
        };
    }

    let empty: Vec<usize> = vec![];

    assert_eq!(
        empty,
        ProductionLHS::new_context_free_elem(FOO).get_all_context_instances(&vec![])
    );

    assert_eq!(
        vec![0],
        ProductionLHS::new_context_free_elem(FOO).get_all_context_instances(&vec![
            pi!(FOO),
            pi!(BAR),
            pi!(BAZ)
        ])
    );

    assert_eq!(
        vec![3],
        ProductionLHS::new_context_free_elem(FOO).get_all_context_instances(&vec![
            pi!(BAR),
            pi!(BAR),
            pi!(BAR3),
            pi!(FOO),
            pi!(BAZ),
            pi!(BAZ),
            pi!(BAZ),
            pi!(BAZ),
            pi!(BAR3)
        ])
    );

    assert_eq!(
        vec![3],
        ProductionLHS::new_with_prefix_list(vec![BAR2, BAR3], "foo".into())
            .get_all_context_instances(&vec![pi!(FOO), pi!(BAR2), pi!(BAR3), pi!(FOO), pi!(BAZ)])
    );

    assert_eq!(
        vec![3],
        ProductionLHS::new_with_prefix_list(vec![BAR2, BAR3], "foo".into())
            .get_all_context_instances(&vec![pi!(FOO), pi!(BAR2), pi!(BAR3), pi!(FOO)])
    );

    assert_eq!(
        vec![4],
        ProductionLHS::new_with_prefix_and_suffix(vec![BAR2, BAR3], "foo".into(), vec![BAR3, BAR2])
            .get_all_context_instances(&vec![
                pi!(BAR2),
                pi!(BAR2),
                pi!(BAR2),
                pi!(BAR3),
                pi!(FOO),
                pi!(BAR3),
                pi!(BAR2),
                pi!(BAZ)
            ])
    );

    assert_eq!(
        vec![1],
        ProductionLHS::new_with_suffix_list(vec![BAR, BAR], "foo".into())
            .get_all_context_instances(&vec![pi!(FOO), pi!(FOO), pi!(BAR), pi!(BAR), pi!(BAR)])
    );

    assert_eq!(
        vec![0],
        ProductionLHS::new_with_suffix_list(vec![BAR, BAR], "foo".into())
            .get_all_context_instances(&vec![pi!(FOO), pi!(BAR), pi!(BAR)])
    );

    assert_eq!(
        vec![3],
        ProductionLHS::new_with_prefix_and_suffix(
            vec![BAR, BAR, BAZ],
            "foo".into(),
            vec![BAZ, BAR, BAR]
        )
        .get_all_context_instances(&vec![
            pi!(BAR),
            pi!(BAR),
            pi!(BAZ),
            pi!(FOO),
            pi!(BAZ),
            pi!(BAR),
            pi!(BAR)
        ])
    );

    assert_eq!(
        vec![4],
        ProductionLHS::new_with_prefix_and_suffix(
            vec![BAR, BAR, BAZ],
            "foo".into(),
            vec![BAZ, BAR, BAR]
        )
        .get_all_context_instances(&vec![
            pi!(FOO),
            pi!(BAR),
            pi!(BAR),
            pi!(BAZ),
            pi!(FOO),
            pi!(BAZ),
            pi!(BAR),
            pi!(BAR)
        ])
    );

    assert_eq!(
        vec![2, 5],
        ProductionLHS::new_with_prefix_and_suffix(vec![BAR], "foo".into(), vec![BAZ])
            .get_all_context_instances(&vec![
                pi!(BAR),
                pi!(BAR),
                pi!(FOO),
                pi!(BAZ),
                pi!(BAR),
                pi!(FOO),
                pi!(BAZ)
            ])
    );

    assert_eq!(
        vec![1, 3],
        ProductionLHS::new_with_prefix_and_suffix(vec![BAR], "foo".into(), vec![BAR])
            .get_all_context_instances(&vec![pi!(BAR), pi!(FOO), pi!(BAR), pi!(FOO), pi!(BAR)])
    );
}

impl<T, I> Debug for ProductionLHS<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // optionally write out prefix.
        if self.prefix.len() > 0 {
            write!(f, "{:?}", self.prefix)?;
        }

        write!(f, "{:?}", self.non_terminal)?;

        // optionally write out suffix.
        if self.suffix.len() > 0 {
            write!(f, "{:?}", self.suffix)?;
        }

        Ok(())
    }
}

impl<T, I> Hash for ProductionLHS<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.prefix.hash(state);
        self.non_terminal.hash(state);
        self.suffix.hash(state);
    }
}

impl<T, I> From<I> for ProductionLHS<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn from(value: I) -> Self {
        Self::new_context_free(value)
    }
}

impl<T, I> PartialOrd for ProductionLHS<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, I> Ord for ProductionLHS<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let shash = hasher.finish();

        let mut hasher = DefaultHasher::new();
        other.hash(&mut hasher);
        let ohash = hasher.finish();

        shash.cmp(&ohash)
    }
}
