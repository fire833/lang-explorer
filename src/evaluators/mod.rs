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

use async_trait::async_trait;

mod compilers;

pub mod karel;

use crate::errors::LangExplorerError;

/// A metric is intended to be usually a real-valued number (or maybe
/// a real-valued vector) that serves as a kind of reward/loss that
/// an Evaluator derives for a particular program that it evaluates.
/// This metric can in turn be fed back to an expander for improving
/// the expander. This is usually going to be in terms of using this
/// loss to compute a gradient for the parameters of the model that
/// the underlying expander uses, and doing gradient descent to optimize
/// the weights.
pub trait Metric: Sized + PartialEq + PartialOrd {}

/// Evaluator is a trait that takes in some program, in the
/// form of a vector of bytes, and returns some kind of metric
/// to be used as reward/error for the specific program that was
/// generated.
#[async_trait]
#[allow(unused)]
pub trait Evaluator {
    type Metric: Metric;

    async fn evaluate(&self, program: Vec<u8>) -> Result<Self::Metric, LangExplorerError>;
}
