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
use futures::{future, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{errors::LangExplorerError, languages::EmbeddingModel};

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
    prompts: Vec<&String>,
    model: EmbeddingModel,
    num_parallel_requests: usize,
) -> Result<Vec<Vec<f32>>, LangExplorerError> {
    let results: Arc<DashMap<usize, Vec<f32>>> = Arc::new(DashMap::new());

    let _ = futures::stream::iter(prompts.iter().enumerate().map(
        async |(idx, prompt)| -> Result<(usize, EmbeddingResult), LangExplorerError> {
            let res = client
                .post(format!("{host}/api/embeddings"))
                .json(&serde_json::json!({
                    "model": model,
                    "prompt": prompt,
                }))
                .send()
                .await?
                .json::<EmbeddingResult>()
                .await?;

            Ok((idx, res))
        },
    ))
    .buffer_unordered(num_parallel_requests)
    .for_each(|result| {
        match result {
            Ok(res) => {
                let data = results.clone();
                data.insert(res.0, res.1.embedding);
            }
            Err(_) => {}
        }

        future::ready(())
    })
    .await;

    let mut results_final: Vec<Vec<f32>> = vec![vec![]; prompts.len()];

    let _ = results
        .iter_mut()
        .map(|kv| results_final[*kv.key()] = kv.value().clone());

    Ok(results_final)
}
