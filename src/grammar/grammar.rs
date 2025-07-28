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
    collections::{BTreeMap, HashMap, HashSet},
    fmt::{Debug, Display},
    io::Write,
};

use sha2::{Digest, Sha256};

use crate::{
    errors::LangExplorerError,
    expanders::GrammarExpander,
    grammar::{
        elem::GrammarElement,
        lhs::ProductionLHS,
        prod::Production,
        program::{InstanceId, ProgramInstance},
        NonTerminal, Terminal,
    },
};

#[derive(Clone)]
pub struct Grammar<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    /// The root symbol of this grammar definition.
    root: I,

    /// The list of productions associated with this grammar.
    productions: BTreeMap<ProductionLHS<T, I>, Production<T, I>>,

    /// Whether or not the grammar is context sensitive. If false,
    /// it is assumed that the grammar is context-free.
    is_context_sensitive: bool,

    /// Canonical name is the unique, human readable name that is
    /// given to this grammar.
    canonical_name: String,
}

impl<T, I> Grammar<T, I>
where
    T: Terminal,
    I: NonTerminal,
{
    pub fn new(root: I, mut productions: Vec<Production<T, I>>, name: String) -> Self {
        let mut map: BTreeMap<ProductionLHS<T, I>, Production<T, I>> = BTreeMap::new();
        let mut is_context_sensitive: bool = false;

        while let Some(p) = productions.pop() {
            let lhs = p.lhs();
            is_context_sensitive |= lhs.is_context_sensitive();
            map.insert(lhs, p);
        }

        Self {
            root,
            productions: map,
            is_context_sensitive,
            canonical_name: name,
        }
    }

    pub fn get_name(&self) -> String {
        return format!("{}_{}", self.canonical_name, &self.generate_uuid()[0..16]);
    }

    pub fn get_productions(&self) -> Vec<&Production<T, I>> {
        return Vec::from_iter(self.productions.values());
    }

    /// Main entrypoint for generating programs from a particular grammar.
    /// Will automatically decide on the expansion regime depending on whether
    /// the grammar is context-free or context-sensitive.
    pub fn generate_program_instance(
        grammar: &Self,
        expander: &mut Box<dyn GrammarExpander<T, I>>,
    ) -> Result<ProgramInstance<T, I>, LangExplorerError> {
        if !grammar.is_context_sensitive {
            return grammar.generate_program_instance_ctx_free(expander);
        } else {
            return grammar.generate_program_instance_ctx_sensitive(expander);
        }
    }

    fn get_next_frontier<'a>(
        frontier: &Vec<&'a ProgramInstance<T, I>>,
    ) -> Option<&'a ProgramInstance<T, I>> {
        for item in frontier.iter() {
            if item.is_non_terminal() {
                return Some(item);
            }
        }

        return None;
    }

    fn generate_program_instance_ctx_sensitive(
        &self,
        expander: &mut Box<dyn GrammarExpander<T, I>>,
    ) -> Result<ProgramInstance<T, I>, LangExplorerError> {
        let mut counter: InstanceId = 1;
        let mut root = ProgramInstance::new(GrammarElement::NonTerminal(self.root.clone()), 1);
        let mut frontier: Vec<&ProgramInstance<T, I>> = vec![&mut root];

        let mut lhs_slots = HashMap::new();
        while let Some(elem) = Grammar::get_next_frontier(&frontier) {
            let nt = elem.get_node();
            lhs_slots.clear();

            for lhs in self.productions.keys() {
                if !lhs.contains(&nt) {
                    continue;
                }

                let instances = lhs.get_all_context_instances(&frontier);
                if instances.len() > 0 {
                    lhs_slots.insert(lhs, instances);
                }
            }

            let (lhs, idx) = expander.choose_lhs_and_slot(&self, &lhs_slots);
            // We literally picked the subset of LHSs that were valid and narrowed
            // down further for this, so it shouldn't fail unless I screw up an
            // expander implementation.
            let prod = self.productions.get(lhs).unwrap();
            let rule = expander.expand_rule(&self, prod);

            // Get the element to be removed and replaced in the frontier.
            let ntp = frontier.remove(idx);

            let mut _children: Vec<ProgramInstance<T, I>> = rule
                .items
                .iter()
                .map(|g| {
                    counter += 1;
                    ProgramInstance::new_with_parent(g.clone(), counter, ntp.get_id())
                })
                .collect();

            // TODO: place children in frontier, and add children to their parent.
            // for child in children.iter_mut().rev() {
            //     frontier.insert(idx, child);
            // }

            // ntp.set_children(children);
        }

        Ok(root)
    }

    fn generate_program_instance_ctx_free(
        &self,
        expander: &mut Box<dyn GrammarExpander<T, I>>,
    ) -> Result<ProgramInstance<T, I>, LangExplorerError> {
        let mut counter: InstanceId = 1;

        let prod = match self
            .productions
            .get(&ProductionLHS::new_context_free(self.root.clone()))
        {
            Some(prod) => prod,
            None => return Err("no root non-terminal/production found".into()),
        };

        match Grammar::generate_program_instance_ctx_free_recursive(
            self,
            prod,
            expander,
            &mut counter,
        ) {
            Ok(program) => Ok(program),
            Err(e) => Err(e),
        }
    }

    fn generate_program_instance_ctx_free_recursive(
        grammar: &Self,
        production: &Production<T, I>,
        expander: &mut Box<dyn GrammarExpander<T, I>>,
        mut counter: &mut InstanceId,
    ) -> Result<ProgramInstance<T, I>, LangExplorerError> {
        let mut program = ProgramInstance::new(
            GrammarElement::NonTerminal(production.non_terminal.non_terminal.clone()),
            *counter,
        );
        let rule = expander.expand_rule(grammar, production);
        let mut children: Vec<ProgramInstance<T, I>> = vec![];
        for item in rule.items.iter() {
            *counter += 1;

            match item {
                GrammarElement::NonTerminal(nt) => {
                    match grammar.productions.get(&ProductionLHS::new_context_free(nt.clone())) // Hack for right now
            {
                Some(prod) => {
                    match Grammar::generate_program_instance_ctx_free_recursive(grammar, prod, expander, &mut counter)  {
                        Ok(instance) => children.push(instance),
                        Err(e) => return Err(e),
                    }
                }
                None => {
                    return Err(format!("non-terminal {:?} not found in productions", nt).into())
                }
            }
                }
                GrammarElement::Epsilon | GrammarElement::Terminal(_) => children.push(
                    ProgramInstance::new_with_parent(item.clone(), *counter, program.get_id()),
                ),
            }
        }

        program.set_children(children);
        Ok(program)
    }

    /// Deprecated: use to generate raw program outputs directly rather
    /// than constructing ASTs.
    #[deprecated(
        note = "Please use generate_program_instance to generate ASTs which can then be serialized to output programs. "
    )]
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
                    match self.productions.get(&ProductionLHS::new_context_free(nt.clone())) // Hack for right now, lol this is never gonna go away isn't it
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

            for item in prod.non_terminal.prefix.iter() {
                set.insert(item);
            }

            for item in prod.non_terminal.suffix.iter() {
                set.insert(item);
            }
        }

        return set;
    }

    /// Generate a unique hash for a given grammar. Essentially it just creates a hash
    /// by deterministically hashing all productions. To do this without implementing orderings
    /// on stuff, I choose to simply traverse the entire tree in in-order traversal, adding every
    /// node to the hash as I go.
    pub fn generate_uuid(&self) -> String {
        let mut hash = Sha256::new();

        self.productions.iter().for_each(|(k, v)| {
            let _ = hash.write(format!("{:?}", k).as_bytes());

            v.items.iter().for_each(|prod| {
                let _ = hash.write(format!("{:?}", prod).as_bytes());
            });
        });

        let digest = hex::encode(hash.finalize());

        digest
    }

    /// Experimental feature to create parsers efficiently
    /// for programs by serializing a grammar into an LALRpop
    /// grammar.
    pub fn generate_lalrpop_parser() -> String {
        "".into()
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
