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

use crate::grammar::{elem::GrammarElement, NonTerminal, Terminal};

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
}

#[derive(Debug)]
enum ContextCheckState {
    Start,
    InPrefix(usize),
    Middle,
    InSuffix(usize),
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
    pub const fn new_context_free(non_terminal: I) -> Self {
        Self {
            prefix: vec![],
            non_terminal,
            suffix: vec![],
        }
    }

    pub fn new_with_prefix_single(prefix: GrammarElement<T, I>, non_terminal: I) -> Self {
        Self::new_with_prefix_list(vec![prefix], non_terminal)
    }

    /// Create a new ProductionLHS with prefix context.
    pub const fn new_with_prefix_list(prefix: Vec<GrammarElement<T, I>>, non_terminal: I) -> Self {
        Self {
            prefix,
            non_terminal,
            suffix: vec![],
        }
    }

    pub fn new_with_suffix_single(suffix: GrammarElement<T, I>, non_terminal: I) -> Self {
        Self::new_with_suffix_list(vec![suffix], non_terminal)
    }

    /// Create a new ProductionLHS with suffix context.
    pub const fn new_with_suffix_list(suffix: Vec<GrammarElement<T, I>>, non_terminal: I) -> Self {
        Self {
            prefix: vec![],
            non_terminal,
            suffix,
        }
    }

    /// Create a new ProductionLHS with both prefix and suffix context.
    pub const fn new_with_prefix_and_suffix(
        prefix: Vec<GrammarElement<T, I>>,
        non_terminal: I,
        suffix: Vec<GrammarElement<T, I>>,
    ) -> Self {
        Self {
            prefix,
            non_terminal,
            suffix,
        }
    }

    /// Returns a list of all GrammarElements of this left hand side.
    pub fn get_all_tokens(&self) -> Vec<GrammarElement<T, I>> {
        let mut tokens = vec![];

        self.prefix.iter().for_each(|p| tokens.push(p.clone()));
        tokens.push(GrammarElement::NonTerminal(self.non_terminal.clone()));
        self.suffix.iter().for_each(|s| tokens.push(s.clone()));

        tokens
    }

    pub fn get_all_context_instances(&self, frontier: &Vec<GrammarElement<T, I>>) -> Vec<usize> {
        let mut instances = vec![];

        let has_prefix = match self.prefix.len() {
            i if i == 0 => false,
            _ => true,
        };

        let has_suffix = match self.suffix.len() {
            i if i == 0 => false,
            _ => true,
        };

        let mut state = ContextCheckState::Start;

        let mut idx = 0;
        while let Some(tok) = frontier.get(idx) {
            let end = idx == frontier.len() - 1;

            match (state, tok, has_prefix, has_suffix, end) {
                (ContextCheckState::Start, _, true, _, true) => return instances,
                (ContextCheckState::Start, elem, true, _, false) => {
                    if *elem == self.prefix[0] {
                        state = ContextCheckState::InPrefix(1);
                    } else {
                        state = ContextCheckState::Start;
                    }
                }
                (ContextCheckState::Start, _, false, true, true) => return instances,
                (ContextCheckState::Start, elem, false, true, false) => {
                    if *elem == self.non_terminal.clone().into() {
                        state = ContextCheckState::Middle;
                    } else {
                        state = ContextCheckState::Start;
                    }
                }
                (ContextCheckState::Start, elem, false, false, true) => {
                    if *elem == self.non_terminal.clone().into() {
                        instances.push(idx);
                    }
                    return instances;
                }
                (ContextCheckState::Start, elem, false, false, false) => {
                    if *elem == self.non_terminal.clone().into() {
                        instances.push(idx);
                    }
                    state = ContextCheckState::Start;
                }

                (ContextCheckState::InPrefix(_), _, _, true, true) => return instances,
                (ContextCheckState::InPrefix(i), elem, _, _, false) => {
                    if i == self.prefix.len() {
                        if *elem == self.non_terminal.clone().into() {
                            state = ContextCheckState::Middle;
                        } else {
                            state = ContextCheckState::Start;
                        }
                    } else if *elem == self.prefix[i] {
                        state = ContextCheckState::InPrefix(i + 1);
                    } else {
                        idx -= 1;
                        state = ContextCheckState::Start;
                    }
                }
                (ContextCheckState::InPrefix(i), elem, _, false, true) => {
                    if i == self.prefix.len() && *elem == self.non_terminal.clone().into() {
                        instances.push(idx - self.prefix.len());
                    }

                    return instances;
                }

                (ContextCheckState::Middle, _, _, true, true) => return instances,
                (ContextCheckState::Middle, elem, false, true, false) => {
                    if *elem == self.suffix[0] {
                        state = ContextCheckState::InSuffix(1);
                    } else if *elem == self.non_terminal.clone().into() {
                        state = ContextCheckState::Middle;
                    } else {
                        state = ContextCheckState::Start;
                    }
                }
                (ContextCheckState::Middle, elem, true, true, false) => {
                    if *elem == self.suffix[0] {
                        state = ContextCheckState::InSuffix(1);
                    } else {
                        state = ContextCheckState::Start;
                    }
                }
                (ContextCheckState::Middle, _, _, false, _) => {
                    instances.push(idx - 1 - self.prefix.len());
                    state = ContextCheckState::Start;
                }

                (ContextCheckState::InSuffix(i), elem, _, _, true) => {
                    if i == self.suffix.len() {
                        instances.push(idx - 1 - self.suffix.len() - self.prefix.len());
                    } else if i == self.suffix.len() - 1 && *elem == self.suffix[i] {
                        instances.push(idx - self.suffix.len() - self.prefix.len());
                    }

                    return instances;
                }
                (ContextCheckState::InSuffix(i), elem, _, _, false) => {
                    if i == self.suffix.len() {
                        instances.push(idx - 1 - self.suffix.len() - self.prefix.len());
                        state = ContextCheckState::Start;
                    } else {
                        if *elem == self.suffix[i] {
                            state = ContextCheckState::InSuffix(i + 1);
                        } else {
                            state = ContextCheckState::Start;
                        }
                    }
                }
            }

            idx += 1;
        }

        instances
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

    let empty: Vec<usize> = vec![];

    assert_eq!(
        empty,
        ProductionLHS::new_context_free_elem(FOO).get_all_context_instances(&vec![])
    );

    assert_eq!(
        vec![0],
        ProductionLHS::new_context_free_elem(FOO).get_all_context_instances(&vec![FOO, BAR, BAR2])
    );

    assert_eq!(
        vec![3],
        ProductionLHS::new_context_free_elem(FOO)
            .get_all_context_instances(&vec![BAR, BAR, BAR3, FOO, BAZ, BAZ, BAZ, BAZ, BAR3])
    );

    assert_eq!(
        vec![1],
        ProductionLHS::new_with_prefix_list(vec![BAR2, BAR3], "foo".into())
            .get_all_context_instances(&vec![FOO, BAR2, BAR3, FOO, BAZ])
    );

    assert_eq!(
        vec![1],
        ProductionLHS::new_with_prefix_list(vec![BAR2, BAR3], "foo".into())
            .get_all_context_instances(&vec![FOO, BAR2, BAR3, FOO])
    );

    assert_eq!(
        vec![2],
        ProductionLHS::new_with_prefix_and_suffix(vec![BAR2, BAR3], "foo".into(), vec![BAR3, BAR2])
            .get_all_context_instances(&vec![BAR2, BAR2, BAR2, BAR3, FOO, BAR3, BAR2, BAZ])
    );

    assert_eq!(
        vec![1],
        ProductionLHS::new_with_suffix_list(vec![BAR, BAR], "foo".into())
            .get_all_context_instances(&vec![FOO, FOO, BAR, BAR, BAR])
    );

    assert_eq!(
        vec![0],
        ProductionLHS::new_with_suffix_list(vec![BAR, BAR], "foo".into())
            .get_all_context_instances(&vec![FOO, BAR, BAR])
    );

    assert_eq!(
        vec![0],
        ProductionLHS::new_with_prefix_and_suffix(
            vec![BAR, BAR, BAZ],
            "foo".into(),
            vec![BAZ, BAR, BAR]
        )
        .get_all_context_instances(&vec![BAR, BAR, BAZ, FOO, BAZ, BAR, BAR])
    );

    assert_eq!(
        vec![1],
        ProductionLHS::new_with_prefix_and_suffix(
            vec![BAR, BAR, BAZ],
            "foo".into(),
            vec![BAZ, BAR, BAR]
        )
        .get_all_context_instances(&vec![FOO, BAR, BAR, BAZ, FOO, BAZ, BAR, BAR])
    );

    assert_eq!(
        vec![1, 4],
        ProductionLHS::new_with_prefix_and_suffix(vec![BAR], "foo".into(), vec![BAZ])
            .get_all_context_instances(&vec![BAR, BAR, FOO, BAZ, BAR, FOO, BAZ])
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
