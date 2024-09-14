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

/// Wrapper trait for all terminal elements to implement.
pub trait Terminal: Sized + Clone + Debug + Hash + Eq + PartialEq + BinarySerialize {}

/// Wrapper trait for all non-terminal elemtns to implement.
pub trait NonTerminal: Sized + Clone + Debug + Hash + Eq + PartialEq {}

#[derive(Clone)]
pub struct Grammar<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    /// The root symbol of this grammar definition.
    root: I,

    /// The list of productions associated with this grammar.
    productions: HashMap<ProductionLHS<T, I>, Production<T, I>>,
}

impl<T, I> Grammar<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    pub fn new(root: I, mut productions: Vec<Production<T, I>>) -> Self {
        let mut map: HashMap<ProductionLHS<T, I>, Production<T, I>> = HashMap::new();

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

        let prod = match self
            .productions
            .get(&ProductionLHS::new_context_free(self.root.clone()))
        {
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
                GrammarElement::NonTerminal(nt) => match self
                    .productions
                    .get(&ProductionLHS::new_context_free(nt.clone())) // Hack for right now
                {
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
    T: Terminal,
    I: NonTerminal,
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
    T: Terminal,
    I: NonTerminal,
{
    /// Reference to the non-terminal that we are using here.
    non_terminal: ProductionLHS<T, I>,

    /// The list of all production rules (ie vectors of vectors of symbols
    /// that can be expanded upon in the grammar expansion process).
    items: Vec<ProductionRule<T, I>>,
}

impl<T, I> Production<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    pub const fn new(non_terminal: ProductionLHS<T, I>, items: Vec<ProductionRule<T, I>>) -> Self {
        Self {
            items,
            non_terminal,
        }
    }

    pub fn get_lhs(&self) -> ProductionLHS<T, I> {
        self.non_terminal.clone()
    }
}

impl<T, I> Debug for Production<T, I>
where
    T: Terminal,
    I: NonTerminal,
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
    prefix: Option<GrammarElement<T, I>>,

    /// non-terminal for the rule.
    non_terminal: I,

    /// optional siffx context for the rule.
    suffix: Option<GrammarElement<T, I>>,
}

impl<T, I> ProductionLHS<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    /// Create a new ProductionLHS with no context, only provide a non-terminal
    /// for expansion.
    pub const fn new_context_free(non_terminal: I) -> Self {
        Self {
            prefix: None,
            non_terminal,
            suffix: None,
        }
    }

    /// Create a new ProductionLHS with prefix context.
    pub const fn new_with_prefix(prefix: GrammarElement<T, I>, non_terminal: I) -> Self {
        Self {
            prefix: Some(prefix),
            non_terminal,
            suffix: None,
        }
    }

    /// Create a new ProductionLHS with suffix context.
    pub const fn new_with_suffix(suffix: GrammarElement<T, I>, non_terminal: I) -> Self {
        Self {
            prefix: None,
            non_terminal,
            suffix: Some(suffix),
        }
    }
}

impl<T, I> Debug for ProductionLHS<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // optionally write out prefix.
        if let Some(prefix) = &self.prefix {
            write!(f, "{:?}", prefix)?;
        }

        write!(f, "{:?}", self.non_terminal)?;

        // optionally write out suffix.
        if let Some(suffix) = &self.suffix {
            write!(f, "{:?}", suffix)?;
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

/// A production rule to use for grammar expansion. Contains a list of
/// GrammarElements that are expanded usually using DFS until only a list of
/// non-terminals remains.
#[derive(Clone)]
pub struct ProductionRule<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    items: Vec<GrammarElement<T, I>>,
}

impl<T, I> ProductionRule<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    pub const fn new(elements: Vec<GrammarElement<T, I>>) -> Self {
        Self { items: elements }
    }
}

impl<T, I> Debug for ProductionRule<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for item in self.items.iter() {
            write!(f, "{:?}", item)?;
        }

        Ok(())
    }
}

/// The atomic elements that comprise the grammar. These can be terminals,
/// which should serialize to a set of bytes (i.e. become valid program code)
/// non-terminals, which are used for
#[derive(Clone, Hash, PartialEq, Eq)]
pub enum GrammarElement<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    Terminal(T),
    NonTerminal(I),
    Epsilon,
}

impl<T, I> GrammarElement<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
}

impl<T, I> Debug for GrammarElement<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Epsilon => write!(f, "ε"),
            Self::NonTerminal(nt) => write!(f, "{:?}", nt),
            Self::Terminal(t) => write!(f, "{:?}", t),
        }
    }
}
