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

pub mod elem;
pub mod grammar;
pub mod lhs;
pub mod prod;
pub mod program;
pub mod rule;

use std::{fmt::Debug, hash::Hash};

#[allow(unused)]
use crate::languages::strings::{nterminal_str, StringValue};

use crate::grammar::elem::GrammarElement;

/// Trait for non-terminals to implement in order to be serialized
/// to an output program.
pub trait BinarySerialize {
    /// Serialize into a Vec.
    #[allow(unused)]
    fn serialize(&self) -> Vec<u8>;

    /// Serialize by appending to the output vector.
    fn serialize_into(&self, output: &mut Vec<u8>);
}

/// Wrapper trait for all terminal elements to implement.
pub trait Terminal: Sized + Clone + Debug + Hash + Eq + PartialEq + Send + BinarySerialize {}

/// Wrapper trait for all non-terminal elements to implement.
pub trait NonTerminal: Sized + Clone + Debug + Hash + Eq + PartialEq + Send {}
