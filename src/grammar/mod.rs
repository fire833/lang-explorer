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

use crate::{errors::LangExplorerError, expanders::GrammarExpander};

/// Trait for non-terminals to implement in order to be serialized
/// to an output program.
pub trait BinarySerialize {
    /// Serialize into a Vec.
    fn serialize(&self) -> Vec<u8>;

    /// Serialize by appending to the output vector.
    fn serialize_into(&self, output: &mut Vec<u8>);
}

#[derive(Clone)]
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

    pub async fn generate_program(
        &self,
        expander: &dyn GrammarExpander<T, I>,
    ) -> Result<Vec<u8>, LangExplorerError> {
        let mut output: Vec<u8> = vec![];

        let prod = match self.productions.get(&self.root) {
            Some(prod) => prod,
            None => return Err("no root non-terminal/production found".into()),
        };

        match self.generate_recursive(&mut output, prod, expander) {
            Ok(_) => Ok(output),
            Err(e) => Err(e),
        }
    }

    fn generate_recursive(
        &self,
        output: &mut Vec<u8>,
        production: &Production<T, I>,
        expander: &dyn GrammarExpander<T, I>,
    ) -> Result<(), LangExplorerError> {
        let rule = expander.expand_rule(&self, production);
        for elem in rule.items.iter() {
            match elem {
                GrammarElement::Terminal(t) => t.serialize_into(output),
                GrammarElement::NonTerminal(nt) => match self.productions.get(nt) {
                    Some(prod) => {
                        if let Err(e) = self.generate_recursive(output, prod, expander) {
                            return Err(e);
                        }
                    }
                    None => {
                        return Err(format!("non-terminal {:?} not found in productions", nt).into())
                    }
                },
                GrammarElement::Epsilon => continue,
            }
        }

        Ok(())
    }
}

impl<T, I> Debug for Grammar<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Entry Symbol: {:?}", self.root)?;

        for (nt, rules) in self.productions.iter() {
            writeln!(f, "{:?}: {:?}", nt, rules)?;
        }

        Ok(())
    }
}

/// Represents all the expansion rules for a particular non-terminal
/// identifier.
#[derive(Clone)]
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
    pub const fn new(non_terminal: I, items: Vec<ProductionRule<T, I>>) -> Self {
        Self {
            items,
            non_terminal,
        }
    }

    pub fn get_lhs(&self) -> I {
        self.non_terminal.clone()
    }
}

impl<T, I> Debug for Production<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, item) in self.items.iter().enumerate() {
            if i != self.items.len() - 1 {
                write!(f, "{:?} |", item)?;
            } else {
                write!(f, " {:?}", item)?;
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct ProductionRule<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    items: Vec<GrammarElement<T, I>>,
}

impl<T, I> ProductionRule<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    pub const fn new(elements: Vec<GrammarElement<T, I>>) -> Self {
        Self { items: elements }
    }
}

impl<T, I> Debug for ProductionRule<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for item in self.items.iter() {
            write!(f, "{:?}", item)?;
        }

        Ok(())
    }
}

#[derive(Clone)]
pub enum GrammarElement<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    Terminal(T),
    NonTerminal(I),
    Epsilon,
}

impl<T, I> GrammarElement<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
}

impl<T, I> Debug for GrammarElement<T, I>
where
    T: Sized + Clone + Debug + BinarySerialize, // Generic terminal type, this will usually be some kind of string or bytes.
    I: Sized + Clone + Debug + Hash + Eq,       // Generic ident non-terminal type.
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Epsilon => write!(f, "ε"),
            Self::NonTerminal(nt) => write!(f, "{:?}", nt),
            Self::Terminal(t) => write!(f, "{:?}", t),
        }
    }
}
