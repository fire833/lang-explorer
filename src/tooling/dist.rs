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

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct Distribution {
    name: String,

    /// The computed moments of the distribution.
    moments: Vec<f32>,

    /// A histogram representation of the distribution.
    histogram: Vec<(f32, usize)>,
}

impl Distribution {
    pub fn from_sample(name: &str, value: &[f32]) -> Self {
        let len = value.len() as f32;
        let mut sum = 0.0_f32;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        value.iter().for_each(|x| {
            let x = *x;
            sum += x;
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        });

        max += 0.001 * (max - min);
        min -= 0.001 * (max - min);

        let n_buckets = 50;
        let bucket_size = (max - min) / n_buckets as f32;
        let mut buckets = vec![];
        let mut val = min;
        for _ in 0..n_buckets {
            buckets.push((val, 0));
            val += bucket_size;
        }

        // Compute histogram
        value.iter().for_each(|x| {
            let x = *x;
            let bucket_index = ((x - min) / bucket_size).floor() as usize;
            buckets[bucket_index] = (buckets[bucket_index].0, buckets[bucket_index].1 + 1);
        });

        let mean = sum / len;

        let variance = value
            .iter()
            .map(|x| {
                let diff = *x - mean;
                diff * diff
            })
            .sum::<f32>()
            / len;

        let sigma = variance.sqrt();

        let skewness = value
            .iter()
            .map(|x| {
                let diff = *x - mean;
                diff * diff * diff
            })
            .sum::<f32>()
            / len
            / (sigma * sigma * sigma);

        let kurtosis = value
            .iter()
            .map(|x| {
                let diff = *x - mean;
                diff * diff * diff * diff
            })
            .sum::<f32>()
            / len
            / (sigma * sigma * sigma * sigma);

        Self {
            name: name.to_string(),
            moments: vec![mean, variance, skewness, kurtosis],
            histogram: buckets,
        }
    }
}
