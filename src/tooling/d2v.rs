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

use std::collections::HashMap;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::errors::LangExplorerError;

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingResult {
    embeddings: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Doc2VecConfig {
    vector_size: u32,
    min_count: u8,
    epochs: u32,
    alpha: f32,
    min_alpha: f32,
    window: u32,
    sample: f32,
    negative: u16,
}

pub(crate) async fn get_embedding_d2v<D: Serialize, W: Serialize>(
    client: &Client,
    host: &String,
    documents: HashMap<D, Vec<W>>,
    config: &Doc2VecConfig,
) -> Result<HashMap<String, Vec<f32>>, LangExplorerError> {
    let body = json!({ "documents": documents, "config": config }).to_string();

    let res = client
        .post(format!("{host}/embed"))
        .body(body)
        .send()
        .await?
        .json::<EmbeddingResult>()
        .await?;

    Ok(res.embeddings)
}
