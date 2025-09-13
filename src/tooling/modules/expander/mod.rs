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

use burn::prelude::Backend;

use crate::tooling::modules::expander::{lin2::Linear2Deep, lin3::Linear3Deep, lin4::Linear4Deep};

pub mod lin2;
pub mod lin3;
pub mod lin4;

pub mod frontier_decision;
pub mod prod_decision_attn;
pub mod prod_decision_fixed;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Activation {
    Sigmoid,
    ReLU,
    LeakyReLU,
    TanH,
}

/// A bit of a hack to allow us to keep an enum for all linear model types.
pub enum LinearModuleWrapper<B: Backend> {
    Linear2(Linear2Deep<B>),
    Linear3(Linear3Deep<B>),
    Linear4(Linear4Deep<B>),
}
