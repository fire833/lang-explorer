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

use crate::{errors::LangExplorerError, languages::EmbeddingModel};

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingResult {
    embedding: Vec<f32>,
}

pub(crate) async fn get_embedding_ollama(
    host: &String,
    prompt: &String,
    model: EmbeddingModel,
) -> Result<Vec<f32>, LangExplorerError> {
    let client = reqwest::Client::new();

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
