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

use crate::grammar::{Grammar, Production};

use super::StringValue;

const POS_OP: &str = "pos";
const FUSE_OP: &str = "fuse";
const SPLIT_OP: &str = "split";
const REORDER_OP: &str = "reorder";

pub fn taco_schedule_grammar() -> Grammar<StringValue, StringValue> {
    Grammar::new(
        "Entrypoint".into(),
        vec![
            Production::new("".into(), vec![]),
            Production::new("".into(), vec![]),
            Production::new("".into(), vec![]),
            Production::new("".into(), vec![]),
        ],
    )
}
