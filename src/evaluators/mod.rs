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

use async_trait::async_trait;

use crate::errors::LangExplorerError;

/// Evaluator is a trait that takes in some program, in the
/// form of a vector of bytes, and returns some kind of metric
/// to be used as reward/error for the specific program that was
/// generated.
#[async_trait]
pub trait Evaluator<M>
where
    M: Sized + Eq + PartialEq + PartialOrd,
{
    async fn evaluate(&self, program: Vec<u8>) -> Result<M, LangExplorerError>;
}
