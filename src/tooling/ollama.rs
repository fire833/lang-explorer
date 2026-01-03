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

use futures::StreamExt;
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

pub(crate) fn get_embedding_ollama_sync(
    client: &reqwest::blocking::Client,
    host: &String,
    prompt: &String,
    model: EmbeddingModel,
) -> Result<Vec<f32>, LangExplorerError> {
    let res = client
        .post(format!("{host}/api/embeddings"))
        .body(format!(
            "{{\"model\": \"{model}\", \"prompt\": \"{prompt}\"}}"
        ))
        .send()?
        .json::<EmbeddingResult>()?;

    Ok(res.embedding)
}

pub(crate) async fn get_embeddings_bulk_ollama(
    client: &Client,
    host: &String,
    prompts: &[String],
    model: EmbeddingModel,
    num_parallel_requests: usize,
) -> Result<Vec<Vec<f32>>, LangExplorerError> {
    let prompts = futures::stream::iter(prompts.iter());
    let res = prompts
        .map(async |prompt| make_request(client, host, &model, prompt, 5).await)
        .buffer_unordered(num_parallel_requests);

    let res = res.collect().await;

    Ok(res)
}

async fn make_request(
    client: &Client,
    host: &String,
    model: &EmbeddingModel,
    prompt: &String,
    mut retries: u8,
) -> Vec<f32> {
    let p = String::from(" ");
    let mut final_prompt = prompt;

    if prompt == "" {
        final_prompt = &p;
    }

    while retries > 0 {
        let res = match client
            .post(format!("{host}/api/embeddings"))
            .json(&serde_json::json!({
                "model": model.to_string(),
                "prompt": final_prompt,
            }))
            .send()
            .await
        {
            Ok(res) => res,
            Err(e) => {
                println!("request error: {}", e);
                retries -= 1;
                continue;
            }
        };

        if res.status() != StatusCode::OK {
            let status = res.status();
            let data = res.text().await.unwrap();
            println!("request returned {status}: {data}");
            retries -= 1;
            continue;
        }

        let emb = match res.json::<EmbeddingResult>().await {
            Ok(emb) => emb,
            Err(e) => {
                println!("deserialization error: {}", e);
                retries -= 1;
                continue;
            }
        };

        return emb.embedding;
    }

    panic!("exceeded maximum retries");
}
