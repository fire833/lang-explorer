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

pub mod program;
pub mod elem;
pub mod rule;
pub mod lhs;
pub mod prod;

use std::{collections::{HashMap, HashSet}, fmt::{Debug, Display}, hash::{ Hash}};

#[allow(unused)]
use crate::languages::strings::{nterminal_str, StringValue};

use crate::{errors::LangExplorerError, expanders::GrammarExpander, grammar::{elem::GrammarElement, lhs::ProductionLHS, prod::Production, program::ProgramInstance}};

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
pub trait Terminal: Sized + Clone + Debug + Hash + Eq + PartialEq + BinarySerialize {}

/// Wrapper trait for all non-terminal elements to implement.
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
            map.insert(p.lhs(), p);
        }

        Self {
            root,
            productions: map,
        }
    }

    pub fn get_productions(&self) -> Vec<&Production<T, I>> {
        return Vec::from_iter(self.productions.values());
    }

    pub fn generate_program_instance(
        &self,
        expander: &mut Box<dyn GrammarExpander<T, I>>,
    ) -> Result<ProgramInstance<T, I>, LangExplorerError> {
        let mut counter: u64 = 1;

        let prod = match self
            .productions
            .get(&ProductionLHS::new_context_free(self.root.clone()))
        {
            Some(prod) => prod,
            None => return Err("no root non-terminal/production found".into()),
        };

        match self.generate_program_instance_recursive(prod, expander, &mut counter) {
            Ok(program) => Ok(program),
            Err(e) => Err(e),
        }
    }

    fn generate_program_instance_recursive(
        &self,
        production: &Production<T, I>,
        expander: &mut Box<dyn GrammarExpander<T, I>>,
        mut counter: &mut u64,
    ) -> Result<ProgramInstance<T, I>, LangExplorerError> {
        let mut program = ProgramInstance::new(GrammarElement::NonTerminal(
            production.non_terminal.non_terminal.clone()
        ), *counter);
        let rule = expander.expand_rule(&self, production);
        let mut children: Vec<ProgramInstance<T, I>> = vec![];
        for item in rule.items.iter() {
            *counter += 1;

            match item {
                GrammarElement::NonTerminal(nt) => {
                    match self.productions.get(&ProductionLHS::new_context_free(nt.clone())) // Hack for right now
            {
                Some(prod) => {
                    match self.generate_program_instance_recursive(prod, expander, &mut counter)  {
                        Ok(instance) => children.push(instance),
                        Err(e) => return Err(e),
                    }                        
                }
                None => {
                    return Err(format!("non-terminal {:?} not found in productions", nt).into())
                }
            }
                }
                GrammarElement::Epsilon | GrammarElement::Terminal(_) => {
                    children.push(ProgramInstance::new_with_parent(item.clone(), *counter, program.get_id()))
                }
            }
        }

        program.set_children(children);
        Ok(program)
    }

    /// Deprecated: use to generate raw program outputs directly rather
    /// than constructing ASTs.
    #[deprecated(note = "Please use generate_program_instance to generate ASTs which can then be serialized to output programs. ")]
    pub fn generate_program(
        &self,
        expander: &mut dyn GrammarExpander<T, I>,
    ) -> Result<Vec<u8>, LangExplorerError> {
        let mut output: Vec<u8> = vec![];

        let prod = match self
            .productions
            .get(&ProductionLHS::new_context_free(self.root.clone()))
        {
            Some(prod) => prod,
            None => return Err("no root non-terminal/production found".into()),
        };

        #[allow(deprecated)]
        match self.generate_recursive(&mut output, prod, expander) {
            Ok(_) => Ok(output),
            Err(e) => Err(e),
        }
    }

    /// Deprecated: use to generate raw program outputs directly rather
    /// than constructing ASTs.
    #[deprecated()]
    fn generate_recursive(
        &self,
        output: &mut Vec<u8>,
        production: &Production<T, I>,
        expander: &mut dyn GrammarExpander<T, I>,
    ) -> Result<(), LangExplorerError> {
        let rule = expander.expand_rule(&self, production);
        for elem in rule.items.iter() {
            match elem {
                GrammarElement::Terminal(t) => t.serialize_into(output),
                GrammarElement::NonTerminal(nt) => {
                    match self.productions.get(&ProductionLHS::new_context_free(nt.clone())) // Hack for right now
                {
                    Some(prod) => {
                        if let Err(e) = self.generate_recursive(output, prod, expander) {
                            return Err(e);
                        }
                    }
                    None => {
                        return Err(format!("non-terminal {:?} not found in productions", nt).into())
                    }
                }
                }
                GrammarElement::Epsilon => continue,
            }
        }

        Ok(())
    }

    /// Retrieve all grammar elements found within this grammar as a set.
    /// Can be useful for creating a static labeling for each element.
    pub fn get_all_nodes(&self) -> HashSet<&GrammarElement<T, I>> {
        let mut set = HashSet::new();

        for prod in self.productions.values() {
            for rule in prod.items.iter() {
                for item in rule.items.iter() {
                    set.insert(item);
                }
            }

            if let Some(prefix) = &prod.non_terminal.prefix {
                for item in prefix.iter() {
                    set.insert(item);
                }
            }

            if let Some(suffix) = &prod.non_terminal.suffix {
                for item in suffix.iter() {
                    set.insert(item);
                }
            }
        }

        return set;
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

/// Displays the grammar in BNF form for easier insertion into the appendices.
impl<T, I> Display for Grammar<T, I> 
where
    T: Terminal,
    I: NonTerminal,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            // Do this twice for ease, make sure we show the root symbol first
            for (nt, rules) in self.productions.iter() {
                if nt.non_terminal != self.root {
                    continue;
                }

                writeln!(f, "<{:?}> ::= {}\n", nt, rules)?;
            }

            for (nt, rules) in self.productions.iter() {
                if nt.non_terminal == self.root {
                    continue;
                }

                writeln!(f, "<{:?}> ::= {}\n", nt, rules)?;
            }

        Ok(())
    }
    }
