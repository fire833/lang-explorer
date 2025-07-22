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

use crate::{
    grammar::{elem::GrammarElement, prod::context_free_production, NonTerminal, Terminal},
    languages::strings::{nterminal_str, terminal_str},
};

#[allow(unused)]
use crate::languages::strings::StringValue;

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

    /// Checks if a LHS is contained within the frontier.
    pub fn check_for_context(&self, frontier: &Vec<GrammarElement<T, I>>) -> Option<usize> {
        enum State {
            Start,
            InPrefix(usize),
            Middle,
            InSuffix(usize),
        }

        let has_prefix = match self.prefix.len() {
            i if i == 0 => false,
            _ => true,
        };

        let has_suffix = match self.suffix.len() {
            i if i == 0 => false,
            _ => true,
        };

        let mut state = State::Start;

        for (idx, tok) in frontier.iter().enumerate() {
            let end = idx == frontier.len() - 1;

            match (state, tok, has_prefix, has_suffix, end) {
                (State::Start, _, true, _, true) => return None,
                (State::Start, elem, true, _, false) => {
                    if *elem == self.prefix[0] {
                        state = State::InPrefix(1);
                    } else {
                        state = State::Start;
                    }
                }
                (State::Start, _, false, true, true) => return None,
                (State::Start, elem, false, true, false) => {
                    if *elem == self.non_terminal.clone().into() {
                        state = State::Middle;
                    } else {
                        state = State::Start;
                    }
                }
                (State::Start, elem, false, false, true) => {
                    if *elem == self.non_terminal.clone().into() {
                        return Some(idx);
                    } else {
                        return None;
                    }
                }
                (State::Start, elem, false, false, false) => {
                    if *elem == self.non_terminal.clone().into() {
                        return Some(idx);
                    } else {
                        state = State::Start;
                    }
                }

                (State::InPrefix(_), _, _, true, true) => return None,
                (State::InPrefix(i), elem, _, _, false) => {
                    if i == self.prefix.len() {
                        if *elem == self.non_terminal.clone().into() {
                            state = State::Middle;
                        } else {
                            state = State::Start;
                        }
                    } else {
                        if *elem == self.prefix[i] {
                            state = State::InPrefix(i + 1);
                        } else {
                            state = State::Start;
                        }
                    }
                }
                (State::InPrefix(i), elem, _, false, true) => {
                    if i == self.prefix.len() && *elem == self.non_terminal.clone().into() {
                        return Some(idx - 1 - self.prefix.len());
                    } else {
                        return None;
                    }
                }
                (State::Middle, _, _, true, true) => return None,
                (State::Middle, elem, true, true, false) => todo!(),
                (State::Middle, elem, true, false, true) => todo!(),
                (State::Middle, elem, true, false, false) => todo!(),
                (State::Middle, elem, false, true, false) => todo!(),
                (State::Middle, elem, false, false, true) => todo!(),
                (State::Middle, elem, false, false, false) => todo!(),
                (State::InSuffix(i), elem, true, true, true) => todo!(),
                (State::InSuffix(i), elem, true, true, false) => todo!(),
                (State::InSuffix(i), elem, true, false, true) => todo!(),
                (State::InSuffix(i), elem, true, false, false) => todo!(),
                (State::InSuffix(i), elem, false, true, true) => todo!(),
                (State::InSuffix(i), elem, false, true, false) => todo!(),
                (State::InSuffix(i), elem, false, false, true) => todo!(),
                (State::InSuffix(i), elem, false, false, false) => todo!(),
            }
        }

        None
    }
}

#[test]
fn test_check_for_context() {
    nterminal_str!(FOO, "foo");
    terminal_str!(BAR, "bar");
    terminal_str!(BAR2, "bar2");
    terminal_str!(BAR3, "bar3");
    nterminal_str!(BAZ, "baz");

    assert_eq!(
        None,
        ProductionLHS::new_context_free_elem(FOO).check_for_context(&vec![])
    );

    assert_eq!(
        Some(0),
        ProductionLHS::new_context_free_elem(FOO).check_for_context(&vec![FOO, BAR, BAR2])
    );

    assert_eq!(
        Some(3),
        ProductionLHS::new_context_free_elem(FOO)
            .check_for_context(&vec![BAR, BAR, BAR3, FOO, BAZ, BAZ, BAZ, BAZ, BAR3])
    );

    assert_eq!(
        Some(1),
        ProductionLHS::new_with_prefix_list(vec![BAR2, BAR3], "foo".into())
            .check_for_context(&vec![FOO, BAR2, BAR3, FOO, BAZ])
    );

    // assert_eq!(
    //     Some(1),
    //     ProductionLHS::new_with_prefix_list(vec![BAR2, BAR3], "foo".into())
    //         .check_for_context(&vec![FOO, BAR2, BAR3, FOO])
    // );
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
