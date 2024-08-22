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

use std::{collections::BTreeMap, fmt::Debug};

pub trait BinarySerialize {
    fn serialize(&self) -> [u8];
}

pub struct Grammar<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug,                   // Generic ident non-terminal type.
{
    /// The root symbol of this grammar definition.
    root: I,

    /// The list of productions associated with this grammar.
    productions: BTreeMap<I, Production<T, I>>,
}

impl<T, I> Grammar<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug,                   // Generic ident non-terminal type.
{
}

pub type ProductionRule<T, I> = Vec<GrammarElement<T, I>>;

/// Represents all the expansion rules for a particular non-terminal
/// identifier.
pub struct Production<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug,                   // Generic ident non-terminal type.
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
    I: Sized + Clone + Debug,                   // Generic ident non-terminal type.
{
    pub fn new(non_terminal: I, items: Vec<ProductionRule<T, I>>) -> Self {
        Self {
            items,
            non_terminal,
        }
    }
}

pub enum GrammarElement<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug,                   // Generic ident non-terminal type.
{
    Terminal(T),
    NonTerminal(I),
    Epsilon,
}
