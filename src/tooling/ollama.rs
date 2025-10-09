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

use std::sync::Arc;

use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use crate::{embedding::EmbeddingModel, errors::LangExplorerError};

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingResult {
    embedding: Vec<f32>,
}

#[allow(unused)]
pub(crate) async fn get_embedding_ollama(
    client: &Client,
    host: &String,
    prompt: &String,
    model: EmbeddingModel,
) -> Result<Vec<f32>, LangExplorerError> {
    let res = client
        .post(format!("{host}/api/embeddings"))
        .body(format!(
            "{{\"model\": \"{model}\", \"prompt\": \"{prompt}\"}}"
        ))
        .send()
        .await?
        .json::<EmbeddingResult>()
        .await?;

    Ok(res.embedding)
}

#[allow(unused)]
pub(crate) async fn get_embeddings_bulk_ollama(
    client: &Client,
    host: &String,
    prompts: &[String],
    model: EmbeddingModel,
    num_parallel_requests: usize,
) -> Result<Vec<Vec<f32>>, LangExplorerError> {
    let results: Arc<DashMap<usize, Vec<f32>>> = Arc::new(DashMap::new());

    let replies = futures::future::join_all(prompts.iter().map(async |prompt| {
        let res = client
            .post(format!("{host}/api/embeddings"))
            .json(&serde_json::json!({
                "model": model,
                "prompt": prompt,
            }))
            .send()
            .await
            .expect("couldn't make request");

        if res.status() != StatusCode::OK {
            panic!("error: invalid reply");
        }

        res.json::<EmbeddingResult>()
            .await
            .expect("couldn't deserialize response")
    }))
    .await;

    let vecs = replies
        .par_iter()
        .map(|reply| reply.embedding.clone())
        .collect();

    Ok(vecs)
}
