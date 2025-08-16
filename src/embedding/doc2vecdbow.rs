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

use std::marker::PhantomData;

use burn::{
    optim::{adaptor::OptimizerAdaptor, AdamW},
    tensor::{backend::AutodiffBackend, Device},
};
use rand_chacha::ChaCha8Rng;

use crate::{
    grammar::{NonTerminal, Terminal},
    tooling::modules::{
        embed::{pvdbow::Doc2VecDBOW, AggregationMethod},
        loss::nsampling::NegativeSampling,
    },
};

pub struct Doc2VecEmbedderDBOW<T: Terminal, I: NonTerminal, B: AutodiffBackend> {
    // Some bullcrap.
    d1: PhantomData<T>,
    d2: PhantomData<I>,

    model: Doc2VecDBOW<B>,
    loss: NegativeSampling<B>,

    device: Device<B>,

    /// RNG stuff.
    rng: ChaCha8Rng,

    optim: OptimizerAdaptor<AdamW, Doc2VecDBOW<B>, B>,

    agg: AggregationMethod,
}
