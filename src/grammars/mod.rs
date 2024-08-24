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

use std::{collections::HashMap, fmt::Debug, hash::Hash};

pub trait BinarySerialize {
    fn serialize(&self) -> [u8];
}

/// A grammar expander is an object that is able to take a
/// current production rule, the whole of the grammar that is
/// being utilized, and is able to spit out a production rule
/// that should be utilized from the list of possible production
/// rules that are implemented by this production.
pub trait GrammarExpander<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize,
    I: Sized + Clone + Debug + Hash + Eq,
{
    fn expand(
        &self,
        grammar: &Grammar<T, I>,
        production: &Production<T, I>,
    ) -> &ProductionRule<T, I>;
}
pub struct Grammar<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    /// The root symbol of this grammar definition.
    root: I,

    /// The list of productions associated with this grammar.
    productions: HashMap<I, Production<T, I>>,
}

impl<T, I> Grammar<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    pub fn new(root: I, mut productions: Vec<Production<T, I>>) -> Self {
        let mut map = HashMap::new();

        while let Some(p) = productions.pop() {
            map.insert(p.get_lhs(), p);
        }

        Self {
            root,
            productions: map,
        }
    }
}

pub type ProductionRule<T, I> = Vec<GrammarElement<T, I>>;

/// Represents all the expansion rules for a particular non-terminal
/// identifier.
pub struct Production<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    /// Reference to the non-terminal that we are using here.
    non_terminal: I,

    /// The list of all production rules (ie vectors of vectors of symbols
    /// that can be expanded upon in the grammar expansion process).
    items: Vec<ProductionRule<T, I>>,
}

impl<T, I> Production<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    pub fn new(non_terminal: I, items: Vec<ProductionRule<T, I>>) -> Self {
        Self {
            items,
            non_terminal,
        }
    }

    pub fn get_lhs(&self) -> I {
        self.non_terminal.clone()
    }
}

pub enum GrammarElement<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    Terminal(T),
    NonTerminal(I),
    Epsilon,
}
