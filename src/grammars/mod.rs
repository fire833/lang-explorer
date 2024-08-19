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

use std::fmt::{Debug, Display};

pub type Grammar<T, I> = Productions<T, I>;

impl<T, I> Grammar<T, I>
where
    T: Sized + Clone, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone, // Generic ident non-terminal type.
{
}

pub struct Productions<T, I>
where
    T: Sized + Clone, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone, // Generic ident non-terminal type.
{
    items: Vec<GrammarElement<T, I>>,
}

impl<T, I> Productions<T, I>
where
    T: Sized + Clone + Debug, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Display, // Generic ident non-terminal type.
{
    pub fn new(items: Vec<GrammarElement<T, I>>) -> Self {
        Self { items }
    }
}

pub enum GrammarElement<T, I>
where
    T: Sized + Clone, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone, // Generic ident non-terminal type.
{
    Terminal(T),
    NonTerminal(I, Productions<T, I>),
    NonTerminalRef(I),
    Epsilon,
}
