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
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    fmt::Debug,
    hash::{Hash, Hasher},
    sync::Arc,
};

use fasthash::{city, FastHasher};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    errors::LangExplorerError,
    expanders::learned::LabelExtractionStrategy,
    grammar::{BinarySerialize, GrammarElement, NonTerminal, Terminal},
    languages::{Feature, ProgramResult},
};

/// Type alias for program instance unique identifiers.
pub type InstanceId = u64;

/// A program instance is a program generated via a particular grammar represented
/// in tree form. This is equivalent to being an AST-representation of a program.
/// So it serves as a wrapper around a general purpose graph. This type is recursively
/// defined.
#[derive(Clone)]
pub struct ProgramInstance<T: Terminal, I: NonTerminal> {
    /// The current node in the tree.
    node: GrammarElement<T, I>,
    /// The list of children nodes.
    children: Vec<ProgramInstance<T, I>>,
    /// A unique identifier for this program instance.
    id: InstanceId,
    /// Optionally the ID of the parent for loopback.
    parent_id: Option<InstanceId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub enum WLKernelHashingOrder {
    #[serde(alias = "self_children_parent")]
    SelfChildrenParentOrdered,
    #[serde(alias = "parent_self_children")]
    ParentSelfChildrenOrdered,
    #[serde(alias = "total_ordered")]
    TotalOrdered,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub enum WLKernelVectorSimilarity {
    #[serde(alias = "l2")]
    Euclidean,
    #[serde(alias = "l1")]
    Manhattan,
}

impl<T: Terminal, I: NonTerminal> ProgramInstance<T, I> {
    /// Create a new program instance. This can be a root of a program tree,
    /// or a subtree.
    pub fn new(node: GrammarElement<T, I>, id: InstanceId) -> Self {
        Self {
            node,
            children: vec![],
            id,
            parent_id: None,
        }
    }

    /// Create a new program instance, but with a parent ID instantiated.
    /// This can be a root of a program tree, or a subtree.
    pub fn new_with_parent(node: GrammarElement<T, I>, id: InstanceId, parent: InstanceId) -> Self {
        Self {
            node,
            children: vec![],
            id,
            parent_id: Some(parent),
        }
    }

    pub(super) fn get_id(&self) -> InstanceId {
        self.id
    }

    pub(super) fn get_node(&self) -> GrammarElement<T, I> {
        self.node.clone()
    }

    pub(super) fn is_non_terminal(&self) -> bool {
        matches!(self.node, GrammarElement::NonTerminal(_))
    }

    /// Add a child node to this current program tree.
    pub(super) fn set_children(&mut self, children: Vec<ProgramInstance<T, I>>) {
        self.children = children;
    }

    /// Insert your children into the frontier with mutable references.
    pub(super) fn add_children_to_frontier<'a>(
        &'a mut self,
        frontier: &mut Vec<&'a mut ProgramInstance<T, I>>,
        idx: usize,
    ) {
        for child in self.children.iter_mut().rev() {
            frontier.insert(idx, child);
        }
    }

    /// Extracts all the "words" for a particular graph using the Weisfeiler-Lehman
    /// graph kernel technique for use within a doc2vec/graph2vec embedding model.
    ///
    /// Reference: https://blog.quarkslab.com/weisfeiler-lehman-graph-kernel-for-binary-function-analysis.html#weisfeiler-lehman%20graph%20kernel_1
    pub(crate) fn extract_words_wl_kernel(
        &self,
        degree: u32,
        ordering: WLKernelHashingOrder,
        dedup: bool,
        sort: bool,
    ) -> Vec<Feature> {
        let nodes = self.get_all_nodes();
        let mut node_features_new: HashMap<&ProgramInstance<T, I>, Feature> = HashMap::new();
        let mut node_features_old: HashMap<&ProgramInstance<T, I>, Feature> = HashMap::new();

        let mut ids: HashMap<Feature, &ProgramInstance<T, I>> = HashMap::new();

        nodes.iter().for_each(|node| {
            ids.insert(node.get_id(), node);
        });

        nodes.iter().for_each(|node| {
            let hash = city::hash64(node.serialize_bytes().as_slice());
            node_features_old.insert(node, hash);
        });

        let mut found_features: Vec<Feature> = node_features_old.values().copied().collect();

        for _ in 0..degree {
            for node in nodes.iter() {
                let self_bytes = node.serialize_bytes();
                let parent_bytes = match node.parent_id {
                    Some(id) => {
                        let parent_feature = match ids.get(&id) {
                            Some(parent) => match node_features_old.get(*parent) {
                                Some(parent_feature) => parent_feature,
                                None => &(0 as Feature),
                            },
                            None => &(0 as Feature),
                        };

                        parent_feature.to_ne_bytes()
                    }
                    None => [0_u8; 8],
                };
                let mut child_bytes: Vec<Feature> = node
                    .children
                    .iter()
                    .map(|child| node_features_old.get(child).unwrap())
                    .copied()
                    .collect();

                let mut hasher = city::Hasher64::new();

                let new_label = match ordering {
                    WLKernelHashingOrder::SelfChildrenParentOrdered => {
                        hasher.write(self_bytes.as_slice());

                        child_bytes.sort();
                        child_bytes
                            .iter()
                            .for_each(|child| hasher.write(&child.to_ne_bytes()));

                        hasher.write(&parent_bytes);

                        hasher.finish()
                    }
                    WLKernelHashingOrder::ParentSelfChildrenOrdered => {
                        hasher.write(&parent_bytes);
                        hasher.write(self_bytes.as_slice());

                        child_bytes.sort();
                        child_bytes
                            .iter()
                            .for_each(|child| hasher.write(&child.to_ne_bytes()));

                        hasher.finish()
                    }
                    WLKernelHashingOrder::TotalOrdered => {
                        // TODO: Implement this ordering.
                        child_bytes.sort();
                        child_bytes
                            .iter()
                            .for_each(|child| hasher.write(&child.to_ne_bytes()));

                        hasher.finish()
                    }
                };

                node_features_new.insert(node, new_label);
            }

            node_features_new
                .values()
                .for_each(|v| found_features.push(*v));

            // Swap out old and new vectors now for the next iteration.
            let cp = node_features_new;
            let cp2 = node_features_old;
            node_features_old = cp;
            node_features_new = cp2;
        }

        if dedup {
            found_features.dedup();
        }

        if sort {
            found_features.sort();
        }

        found_features
    }

    /// Returns the current program in edge-list form. This will be useful for importing
    /// into other graph processing software like networkx.
    pub(crate) fn get_edge_list(&self) -> Vec<(InstanceId, InstanceId)> {
        let mut edges = vec![];

        if self.children.is_empty() {
            return edges;
        }

        for child in self.children.iter() {
            edges.push((self.id, child.id));
        }

        for child in self.children.iter() {
            edges.append(&mut child.get_edge_list());
        }

        edges
    }

    pub(crate) fn serialize_bytes(&self) -> Vec<u8> {
        match &self.node {
            GrammarElement::Terminal(t) => t.serialize(),
            // Hacky as all molasses, but works for now.
            GrammarElement::NonTerminal(nt) => Vec::from(format!("{:?}", nt).as_bytes()),
            GrammarElement::Epsilon => Vec::from(b"epsilon"),
        }
    }

    /// Essentially run BFS on the graph to get all nodes.
    pub(crate) fn get_all_nodes(&self) -> Vec<&ProgramInstance<T, I>> {
        let mut nodes = vec![];

        // Since these will all be acyclic trees, we don't need to worry about
        // keeping a visited set. Haha I say that now.
        // let mut visited: HashSet<&ProgramInstance<T, I>> = HashSet::new();
        let mut queue: VecDeque<&ProgramInstance<T, I>> = VecDeque::new();
        queue.push_back(self);

        while let Some(node) = queue.pop_front() {
            nodes.push(node);

            for child in node.children.iter() {
                queue.push_back(child);
            }
        }

        nodes
    }

    /// Create a slightly perturbed program.
    pub(crate) fn perturb_program(&self, rng: &mut ChaCha8Rng) -> ProgramInstance<T, I> {
        let mut new = self.clone();
        let mut curr = &mut new;
        let mut all_t = false;
        while !all_t {
            all_t = true;

            curr.children.iter().for_each(|child| {
                if child.is_non_terminal() {
                    all_t = false;
                }
            });

            if all_t {
                break;
            }

            let len = curr.children.len();
            let n = curr
                .children
                .get_mut(rng.random::<u64>() as usize % len)
                .unwrap();

            curr = n;
        }

        curr.children
            .remove(rng.random::<u64>() as usize % curr.children.len());

        // Optionally replace with a new child node.
        // if rng.random::<bool>() {}

        new
    }

    pub(crate) fn get_all_nodes_exclude_children(&self) -> Vec<&ProgramInstance<T, I>> {
        let mut programs = vec![self];

        for node in self.get_all_nodes() {}

        programs
    }

    /// Serialize the current program instance into a graphviz graph string.
    pub(crate) fn serialize_to_graphviz(&self) -> String {
        let mut s: String = "digraph { ".into();

        let mut queue = VecDeque::new();
        queue.push_back(self);

        while let Some(node) = queue.pop_front() {
            let color: &str = match node.node {
                GrammarElement::Terminal(_) => "red",
                GrammarElement::NonTerminal(_) => "blue",
                GrammarElement::Epsilon => "yellow",
            };

            s.push_str(&format!(
                "n{} [color={color}, label=\"{:?}\"]; ",
                node.id, node.node,
            ));

            for child in node.children.iter() {
                queue.push_back(child);
                s.push_str(&format!("n{} -> n{}; ", node.id, child.id));
            }
        }

        s.push_str(" }");

        s
    }

    pub(crate) fn to_result(
        &self,
        return_features: bool,
        return_edge_lists: bool,
        return_graphviz: bool,
        is_complete: bool,
        label_extraction: &LabelExtractionStrategy,
    ) -> Result<ProgramResult, LangExplorerError> {
        let mut res = ProgramResult::new();

        if return_features {
            match label_extraction {
                LabelExtractionStrategy::WLKernel {
                    iterations,
                    order,
                    dedup,
                    sort,
                } => {
                    res.set_features(self.extract_words_wl_kernel(
                        *iterations,
                        order.clone(),
                        *dedup,
                        *sort,
                    ));
                }
                LabelExtractionStrategy::CodePaths {} => todo!(),
            }
        }

        if return_edge_lists {
            res.set_edge_list(self.get_edge_list());
        }

        if return_graphviz {
            res.set_graphviz(self.serialize_to_graphviz());
        }

        res.set_is_partial(!is_complete);

        if is_complete {
            match String::from_utf8(self.serialize()) {
                Ok(data) => res.set_program(data),
                Err(e) => return Err(e.into()),
            }
        } else {
            res.set_program(self.to_string());
        }

        Ok(res)
    }

    /// Extract a rooted sub-program from this program instance of degree d.
    /// Used for graph2vec implementation, where we need all rooted subgraphs
    /// of degree d of a particular graph.
    #[deprecated()]
    fn get_subgraph(&self, degree: u32) -> ProgramInstance<T, I> {
        if degree == 0 {
            ProgramInstance::new(self.node.clone(), self.id)
        } else {
            let mut newchildren = vec![];
            for child in self.children.iter() {
                let subgraph = child.get_subgraph(degree - 1);
                newchildren.push(subgraph);
            }

            let mut prog = ProgramInstance::new(self.node.clone(), self.id);
            prog.children = newchildren;

            prog
        }
    }

    /// Returns all rooted subgraphs for the provided graph of degree d.
    #[deprecated()]
    pub fn get_all_subgraphs(&self, degree: u32) -> Vec<ProgramInstance<T, I>> {
        #[allow(deprecated)]
        let mut subgraphs = vec![self.get_subgraph(degree)];
        for child in self.children.iter() {
            subgraphs.append(&mut child.get_all_subgraphs(degree));
        }

        subgraphs
    }

    pub fn simrank_similarity(&self, other: &ProgramInstance<T, I>, c: f64, depth: u32) -> f64 {
        if self.node == other.node && self.children.is_empty() && other.children.is_empty() {
            return 1.0;
        }

        if self.children.is_empty() || other.children.is_empty() || depth == 0 {
            return 0.0;
        }

        let mut sim = 0.0;

        for v in self.children.iter() {
            for w in other.children.iter() {
                sim += v.simrank_similarity(w, c, depth - 1);
            }
        }

        let denom = (self.children.len() * other.children.len()) as f64;

        if denom == 0.0 {
            0.0
        } else {
            (c * sim) / denom
        }
    }

    pub fn wl_test(
        &self,
        other: &ProgramInstance<T, I>,
        ordering: WLKernelHashingOrder,
        similarity: WLKernelVectorSimilarity,
        iterations: u32,
        dedup: bool,
        sort: bool,
    ) -> f32 {
        let self_features = self.extract_words_wl_kernel(iterations, ordering.clone(), dedup, sort);
        let other_features = other.extract_words_wl_kernel(iterations, ordering, dedup, sort);

        // Mapping between a feature and (self count, other count).
        let mut set: BTreeMap<u64, (u32, u32)> = BTreeMap::new();

        self_features.iter().for_each(|f| {
            let entry = set.entry(*f).or_insert((0, 0));
            entry.0 += 1;
        });

        other_features.iter().for_each(|f| {
            let entry = set.entry(*f).or_insert((0, 0));
            entry.1 += 1;
        });

        match similarity {
            WLKernelVectorSimilarity::Euclidean => (set
                .iter()
                .map(|entry| (entry.1 .0 as i32 - entry.1 .1 as i32).pow(2) as u32)
                .sum::<u32>() as f32)
                .sqrt(),
            WLKernelVectorSimilarity::Manhattan => set
                .iter()
                .map(|entry| (entry.1 .0 as i32).abs_diff(entry.1 .1 as i32))
                .sum::<u32>() as f32,
        }
    }
}

#[test]
fn test_extract_words_wl_kernel() {
    use crate::languages::strings::{nterminal_str, StringValue};

    nterminal_str!(FOO, "foo");
    nterminal_str!(BAR, "bar");
    nterminal_str!(BAZ, "baz");
    nterminal_str!(BUZZ, "buzz");

    let mut program = ProgramInstance::new(FOO, 1);
    program.set_children(vec![
        ProgramInstance::new(BAR, 2),
        ProgramInstance::new(BAZ, 3),
        ProgramInstance::new(FOO, 4),
        ProgramInstance::new(BUZZ, 5),
    ]);

    let mut words = program.extract_words_wl_kernel(
        4,
        WLKernelHashingOrder::SelfChildrenParentOrdered,
        false,
        false,
    );

    let mut wordsdedup = program.extract_words_wl_kernel(
        4,
        WLKernelHashingOrder::SelfChildrenParentOrdered,
        true,
        false,
    );

    words.sort();
    wordsdedup.sort();

    println!("{:?}", words);
    println!("{:?}", wordsdedup);
}

impl<T: Terminal, I: NonTerminal> BinarySerialize for ProgramInstance<T, I> {
    fn serialize(&self) -> Vec<u8> {
        let mut vec: Vec<u8> = vec![];

        match &self.node {
            GrammarElement::Terminal(t) => {
                let mut c = t.serialize();
                vec.append(&mut c);
            }
            GrammarElement::NonTerminal(_) => {
                for child in self.children.iter() {
                    let mut c = child.serialize();
                    vec.append(&mut c);
                }
            }
            GrammarElement::Epsilon => {}
        }

        vec
    }

    fn serialize_into(&self, output: &mut Vec<u8>) {
        let mut vec = self.serialize();
        output.append(&mut vec);
    }
}

/// Implement partial equality for program instance.
/// Since we just care about node equality, just compare
/// the grammar elements within, don't worry about
/// children for now. This will probably bite me later
/// but lets just deal with it.
impl<T: Terminal, I: NonTerminal> PartialEq for ProgramInstance<T, I> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl<T: Terminal, I: NonTerminal> Eq for ProgramInstance<T, I> {}

impl<T: Terminal, I: NonTerminal> Hash for ProgramInstance<T, I> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        // This will probably also come back to bite me, will most
        // likely need to be changed because I can't keep hashing the
        // entirety of these massive trees.
        //
        // Follow up: Don't really know what to do about this right now.
        //
        // self.children.hash(state);
    }
}

impl<T: Terminal, I: NonTerminal> Debug for ProgramInstance<T, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.node)
    }
}

impl<T: Terminal, I: NonTerminal> ToString for ProgramInstance<T, I> {
    fn to_string(&self) -> String {
        let mut res = format!("{:?}", self.node);
        for child in self.children.iter() {
            res.push_str(&child.to_string());
        }

        res
    }
}

impl<T: Terminal, I: NonTerminal> From<Arc<ProgramInstance<T, I>>> for ProgramInstance<T, I> {
    fn from(value: Arc<ProgramInstance<T, I>>) -> Self {
        Self {
            node: value.node.clone(),
            children: value.children.clone(),
            id: value.id,
            parent_id: value.parent_id,
        }
    }
}
