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

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::languages::Feature;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub enum VectorSimilarity {
    #[serde(rename = "l2")]
    Euclidean,
    #[serde(rename = "l1")]
    Manhattan,
}

pub fn wl_test(vec1: &[Feature], vec2: &[Feature], similarity: VectorSimilarity) -> f32 {
    // Mapping between a feature and (self count, other count).
    let mut set: FxHashMap<u64, (u32, u32)> = FxHashMap::default();

    vec1.iter().for_each(|f| {
        let entry = set.entry(*f).or_insert((0, 0));
        entry.0 += 1;
    });

    vec2.iter().for_each(|f| {
        let entry = set.entry(*f).or_insert((0, 0));
        entry.1 += 1;
    });

    match similarity {
        VectorSimilarity::Euclidean => (set
            .iter()
            .map(|entry| (entry.1 .0 as i32 - entry.1 .1 as i32).pow(2) as u32)
            .sum::<u32>() as f32)
            .sqrt(),
        VectorSimilarity::Manhattan => set
            .iter()
            .map(|entry| (entry.1 .0 as i32).abs_diff(entry.1 .1 as i32))
            .sum::<u32>() as f32,
    }
}

pub fn vector_similarity(vec1: &[f32], vec2: &[f32], similarity: VectorSimilarity) -> f32 {
    match similarity {
        VectorSimilarity::Euclidean => vec1
            .iter()
            .zip(vec2.iter())
            .map(|(a, b)| (*a - *b).powf(2.0))
            .sum::<f32>()
            .sqrt(),
        VectorSimilarity::Manhattan => vec1
            .iter()
            .zip(vec2.iter())
            .map(|(a, b)| (*a - *b).abs())
            .sum::<f32>(),
    }
}
