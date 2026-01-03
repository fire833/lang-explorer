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

use std::{
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    str::FromStr,
};

use burn::prelude::Backend;
use bytes::Bytes;
use lang_explorer::{
    expanders::ExpanderWrapper,
    experiments::generate::GenerateInput,
    languages::LanguageWrapper,
    tooling::api::{
        catchall_handler, health_handler, invalid_request, ErrorMessage, OpenApiExtensions,
    },
};
use lang_explorer::{experiments::generate::GenerateOutput, tooling::api::openapi_handler};
use utoipa::OpenApi;
use warp::{
    http::StatusCode,
    reply::{Json, WithStatus},
    Filter, Rejection,
};

#[derive(OpenApi)]
#[openapi(
    paths(generate),
    components(
        schemas(LanguageWrapper, ExpanderWrapper, GenerateOutput, GenerateInput),
        responses()
    ),
    info(
        description = "OpenAPI specification for the Language Explorer API.",
        title = "Language Explorer API",
        version = "0.2.0",
        contact(name = "Kendall Tauser", email = "kttpsy@gmail.com"),
        license(name = "GPL2", identifier = "GPL2")
    )
)]
struct ExplorerAPIDocs;

impl OpenApiExtensions for ExplorerAPIDocs {}

pub(super) async fn start_server<B: Backend>(
    addr: &str,
    port: u16,
    models_dir: String,
    ollama_host: String,
    d2v_host: String,
    output_dir: String,
) {
    let _model_filter = warp::any().map(move || models_dir.clone());
    let _ollama_filter = warp::any().map(move || ollama_host.clone());
    let _d2v_filter = warp::any().map(move || d2v_host.clone());
    let _output_filter = warp::any().map(move || output_dir.clone());

    // let generate2 = warp::post()
    //     .and(warp::path!(
    //         "v2" / "generate" / LanguageWrapper / ExpanderWrapper
    //     ))
    //     .and(warp::path::end())
    //     .and(warp::body::bytes())
    //     .and(model_filter.clone())
    //     .and(ollama_filter.clone())
    //     .and(d2v_filter.clone())
    //     .and(output_filter.clone())
    //     .and_then(generate::<B>);

    let cors = warp::cors()
        .allow_any_origin()
        .allow_header("Content-Type")
        .allow_header("Accept")
        .allow_header("User-Agent")
        .allow_methods(vec!["POST", "GET"]);

    let routes = health_handler()
        // generate2
        .or(openapi_handler::<ExplorerAPIDocs>())
        .or(catchall_handler())
        .with(cors);

    warp::serve(routes)
        .run(SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::from_str(addr).expect("invalid bind ip address"),
            port,
        )))
        .await;
}

#[utoipa::path(
    get, path = "/v2/generate/{language}/{expander}", 
    request_body = GenerateInput,
    responses(
        (status = 200, description = "Successfully generated code.", body = GenerateOutput),
        (status = 400, description = "Invalid request was made to the server.", body = ErrorMessage)
    ),
    params(
        ("language" = LanguageWrapper, Path, description = "The language to use."),
        ("expander" = ExpanderWrapper, Path, description = "The expander to utilize."),
    )
)]
#[allow(unused)]
async fn generate<B: Backend>(
    language: LanguageWrapper,
    expander: ExpanderWrapper,
    body: Bytes,
    models_dir: String,
    ollama_host: String,
    d2v_host: String,
    output_dir: String,
) -> Result<WithStatus<Json>, Rejection> {
    let params = match serde_json::from_slice::<GenerateInput>(&body) {
        Ok(p) => p,
        Err(e) => return invalid_request(e.to_string()),
    };

    match params
        .execute::<B>(
            language,
            expander,
            models_dir,
            ollama_host,
            d2v_host,
            output_dir,
        )
        .await
    {
        Ok(resp) => {
            let code = StatusCode::OK;
            Ok(warp::reply::with_status(warp::reply::json(&resp), code))
        }
        Err(e) => invalid_request(e.to_string()),
    }
}
