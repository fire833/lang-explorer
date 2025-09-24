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

/// The different strategies for choosing the next expansion rule
/// given the probability distribution from the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Choose the highest probability expansion.
    HighestProb,

    /// Choose the lowest probability distribution,
    /// you probably don't want to do this if you
    /// care about your output.
    LowestProb,
}

/// The different strategies for normalizing the output logits
/// given by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationStrategy {
    Softmax,
    LogSoftmax,
}
