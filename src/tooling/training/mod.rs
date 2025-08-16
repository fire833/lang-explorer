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

use burn::config::Config;

/// Common training parameters used by most (if not all) modules.
#[derive(Config, Debug)]
pub struct TrainingParams {
    /// The size of the batches fed through the model.
    #[config(default = 256)]
    pub batch_size: usize,
    /// Number of epochs to train.
    #[config(default = 10)]
    pub n_epochs: usize,
    /// The learning rate
    #[config(default = 0.001)]
    pub learning_rate: f64,
    /// The seed.
    #[config(default = 10)]
    pub seed: u64,
}
