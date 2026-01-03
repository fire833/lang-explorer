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

use burn::config::Config;
use utoipa::ToSchema;

pub mod code2vec;
pub mod gat;
mod gathead;
mod gatlayer;
pub mod gcn;
mod gcnconv;
pub mod gnn;
pub mod pvdbow;
pub mod pvdm;
pub mod wvcbow;
pub mod wvsg;

#[derive(Debug, Config, ToSchema)]
pub enum AggregationMethod {
    Average,
    Sum,
}

impl Default for AggregationMethod {
    fn default() -> Self {
        AggregationMethod::Average
    }
}
