/*
*	Copyright (C) 2026 Kendall Tauser
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

use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
    tensor::{Float, Tensor},
};

use crate::{
    grammar::program::ProgramInstance,
    tooling::modules::{embed::gathead::GATHead, expander::Activation},
};

#[derive(Debug, Config)]
pub struct GATLayerConfig {}

impl GATLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GATLayer<B> {
        GATLayer { heads: vec![] }
    }
}

#[derive(Debug, Module)]
pub struct GATLayer<B: Backend> {
    heads: Vec<GATHead<B>>,
}

impl<B: Backend> GATLayer<B> {
    pub fn forward(
        &self,
        node_features: Tensor<B, 3, Float>,
        programs: &Vec<&ProgramInstance>,
        activation: Activation,
    ) -> Tensor<B, 3, Float> {
        todo!()
    }
}

#[test]
fn test_forward() {}
