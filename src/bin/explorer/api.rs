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

use std::{
    convert::Infallible,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    str::FromStr,
};

use burn::prelude::Backend;
use lang_explorer::languages::{GenerateResults, GenerateResultsV2};
use lang_explorer::{
    expanders::ExpanderWrapper,
    languages::{GenerateParams, LanguageWrapper},
};
use serde::{Deserialize, Serialize};
use utoipa::{OpenApi, ToSchema};
use warp::{
    http::StatusCode,
    hyper::body::Bytes,
    reject::Rejection,
    reply::{Json, WithStatus},
    Filter,
};

#[derive(OpenApi)]
#[openapi(
    paths(generate, generate_legacy),
    components(
        schemas(
            LanguageWrapper,
            ExpanderWrapper,
            GenerateResultsV2,
            GenerateParams,
            GenerateResults
        ),
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

pub(super) async fn start_server<B: Backend>(addr: &str, port: u16, models_dir: String) {
    let model_filter = warp::any().map(move || models_dir.clone());

    #[allow(deprecated)]
    let generate1 = warp::post()
        .and(warp::path!(
            "v1" / "generate" / LanguageWrapper / ExpanderWrapper
        ))
        .and(warp::path::end())
        .and(warp::body::bytes())
        .and(model_filter.clone())
        .and_then(generate_legacy);

    let generate2 = warp::post()
        .and(warp::path!(
            "v2" / "generate" / LanguageWrapper / ExpanderWrapper
        ))
        .and(warp::path::end())
        .and(warp::body::bytes())
        .and(model_filter.clone())
        .and_then(generate::<B>);

    let health = warp::get()
        .and(warp::path!("readyz"))
        .or(warp::path!("livez"))
        .and(warp::path::end())
        .map(|_| ready_ok());

    let openapi = warp::get()
        .and(warp::path!("swagger.json"))
        .or(warp::path!("openapi.json"))
        .or(warp::path!("api-docs"))
        .and(warp::path::end())
        .map(|_| ExplorerAPIDocs::api_docs_reply());

    let any_handler = warp::any().map(|| not_found());

    let cors = warp::cors()
        .allow_any_origin()
        .allow_header("Content-Type")
        .allow_header("Accept")
        .allow_methods(vec!["POST", "GET"]);

    let routes = generate1
        .or(generate2)
        .or(health)
        .or(openapi)
        .or(any_handler)
        .with(cors);

    warp::serve(routes)
        .run(SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::from_str(addr).expect("invalid bind ip address"),
            port,
        )))
        .await;
}

#[utoipa::path(
    get, path = "/v1/generate/{language}/{expander}", 
    request_body = GenerateParams,
    responses(
        (status = 200, description = "Successfully generated code.", body = GenerateResults),
        (status = 400, description = "Invalid request was made to the server.", body = ErrorMessage)
    ),
    params(
        ("language" = LanguageWrapper, Path, description = "The language to use."),
        ("expander" = ExpanderWrapper, Path, description = "The expander to utilize."),
    )
)]
#[deprecated()]
async fn generate_legacy(
    language: LanguageWrapper,
    expander: ExpanderWrapper,
    body: Bytes,
    _models_dir: String,
) -> Result<impl warp::Reply, Infallible> {
    let params = match serde_json::from_slice::<GenerateParams>(&body) {
        Ok(p) => p,
        Err(e) => return Ok(invalid_request(e.to_string())),
    };

    #[allow(deprecated)]
    match params.execute_legacy(language, expander).await {
        Ok(resp) => Ok(warp::reply::with_status(
            warp::reply::json(&resp),
            StatusCode::OK,
        )),
        Err(e) => Ok(invalid_request(e.to_string())),
    }
}

#[utoipa::path(
    get, path = "/v2/generate/{language}/{expander}", 
    request_body = GenerateParams,
    responses(
        (status = 200, description = "Successfully generated code.", body = GenerateResultsV2),
        (status = 400, description = "Invalid request was made to the server.", body = ErrorMessage)
    ),
    params(
        ("language" = LanguageWrapper, Path, description = "The language to use."),
        ("expander" = ExpanderWrapper, Path, description = "The expander to utilize."),
    )
)]
async fn generate<B: Backend>(
    language: LanguageWrapper,
    expander: ExpanderWrapper,
    body: Bytes,
    models_dir: String,
) -> Result<impl warp::Reply, Infallible> {
    let params = match serde_json::from_slice::<GenerateParams>(&body) {
        Ok(p) => p,
        Err(e) => return Ok(invalid_request(e.to_string())),
    };

    match params.execute::<B>(language, expander, models_dir).await {
        Ok(resp) => Ok(warp::reply::with_status(
            warp::reply::json(&resp),
            StatusCode::OK,
        )),
        Err(e) => Ok(invalid_request(e.to_string())),
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct ErrorMessage {
    code: u16,
    message: String,
}

impl ErrorMessage {
    #[allow(unused)]
    fn new(code: u16, message: &str) -> Self {
        Self {
            code,
            message: message.to_string(),
        }
    }

    fn new_from_string(code: u16, message: String) -> Self {
        Self { code, message }
    }
}

#[allow(unused)]
fn not_found() -> WithStatus<Json> {
    let code = StatusCode::NOT_FOUND;
    warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new(code.into(), "resource not found")),
        code,
    )
}

fn ready_ok() -> WithStatus<Json> {
    let code = StatusCode::OK;
    warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new(code.into(), "application is ready")),
        code,
    )
}

#[allow(unused)]
fn invalid_authorization() -> WithStatus<Json> {
    let code = StatusCode::UNAUTHORIZED;
    warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new(
            code.into(),
            "invalid authorization credentials provided",
        )),
        code,
    )
}

#[allow(unused)]
async fn invalid_request_rejection(rej: Rejection) -> Result<impl warp::Reply, Infallible> {
    let code = StatusCode::BAD_REQUEST;
    println!("invalid request made: {:?}", rej);
    Ok(warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new_from_string(
            code.into(),
            format!("invalid request: {:?}", rej),
        )),
        code,
    ))
}

fn invalid_request(err: String) -> WithStatus<Json> {
    let code = StatusCode::BAD_REQUEST;
    println!("invalid request made: {}", err);
    warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new_from_string(
            code.into(),
            format!("invalid request: {}", err),
        )),
        code,
    )
}

#[allow(unused)]
fn internal_error(err: String) -> WithStatus<Json> {
    let code = StatusCode::INTERNAL_SERVER_ERROR;
    warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new_from_string(
            code.into(),
            format!("unable to execute request: {}", err),
        )),
        code,
    )
}

pub trait OpenApiExtensions: OpenApi {
    fn api_docs_reply() -> WithStatus<Json> {
        warp::reply::with_status(warp::reply::json(&Self::openapi()), StatusCode::OK)
    }

    #[allow(unused)]
    fn api_docs_print() {
        println!(
            "{}",
            Self::openapi().to_pretty_json().unwrap_or("".to_string())
        );
    }
}
