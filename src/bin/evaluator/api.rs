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
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    str::FromStr,
};

use bytes::Bytes;
use lang_explorer::tooling::api::{
    catchall_handler, health_handler, openapi_handler, ready_ok, ErrorMessage, OpenApiExtensions,
};
use serde::{Deserialize, Serialize};
use utoipa::{OpenApi, ToSchema};
use warp::{
    reply::{Json, WithStatus},
    Filter, Rejection,
};

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct KarelEvaluationResult {
    pub success: bool,
    pub score: f64,
    pub execution_time_ms: u64,
    pub error_message: Option<String>,
    pub output: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct TacoschedEvaluationResult {
    pub success: bool,
    pub validation_errors: Vec<String>,
    pub execution_time_ms: u64,
}

#[derive(OpenApi)]
#[openapi(
    paths(evaluate_karel, evaluate_tacosched),
    components(
        schemas(
            // ProgramInstance,
            KarelEvaluationResult,
            TacoschedEvaluationResult,
            ErrorMessage
        ),
        responses()
    ),
    info(
        description = "OpenAPI specification for the Evaluation API.",
        title = "Evaluation API",
        version = "1.0.0",
        contact(name = "Kendall Tauser", email = "kttpsy@gmail.com"),
        license(name = "GPL2", identifier = "GPL2")
    )
)]
struct EvaluationAPIDocs;

impl OpenApiExtensions for EvaluationAPIDocs {}

pub async fn start_server(addr: &str, port: u16) {
    // Karel evaluation endpoint
    let karel_route = warp::post()
        .and(warp::path!("v1" / "evaluate" / "karel"))
        .and(warp::path::end())
        .and(warp::body::bytes())
        .and_then(evaluate_karel);

    // Tacosched evaluation endpoint
    let tacosched_route = warp::post()
        .and(warp::path!("v1" / "evaluate" / "tacosched"))
        .and(warp::path::end())
        .and(warp::body::bytes())
        .and_then(evaluate_tacosched);

    // CORS configuration
    let cors = warp::cors()
        .allow_any_origin()
        .allow_header("Content-Type")
        .allow_header("Accept")
        .allow_header("User-Agent")
        .allow_methods(vec!["POST", "GET"]);

    // Combine all routes
    let routes = karel_route
        .or(tacosched_route)
        .or(health_handler())
        .or(openapi_handler::<EvaluationAPIDocs>())
        .or(catchall_handler())
        .with(cors);

    // Start the server
    warp::serve(routes)
        .run(SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::from_str(addr).expect("invalid bind ip address"),
            port,
        )))
        .await;
}

#[utoipa::path(
    post,
    path = "/v1/evaluate/karel",
    // request_body = ProgramInstance,
    responses(
        (status = 200, description = "Successfully evaluated Karel program.", body = KarelEvaluationResult),
        (status = 400, description = "Invalid request was made to the server.", body = ErrorMessage)
    ),
)]
async fn evaluate_karel(_body: Bytes) -> Result<WithStatus<Json>, Rejection> {
    // let program_instance = match serde_json::from_slice::<ProgramInstance>(&body) {
    //     Ok(p) => p,
    //     Err(e) => return invalid_request(e.to_string()).await,
    // };

    // // TODO: Implement Karel evaluation logic here
    // match execute_karel_evaluation(&program_instance, &config_dir).await {
    //     Ok(result) => {
    //         let code = StatusCode::OK;
    //         Ok(warp::reply::with_status(warp::reply::json(&result), code))
    //     }
    //     Err(e) => invalid_request(e.to_string()).await,
    // }

    ready_ok()
}

#[utoipa::path(
    post,
    path = "/v1/evaluate/tacosched",
    // request_body = ProgramInstance,
    responses(
        (status = 200, description = "Successfully evaluated Tacosched program.", body = TacoschedEvaluationResult),
        (status = 400, description = "Invalid request was made to the server.", body = ErrorMessage)
    ),
)]
async fn evaluate_tacosched(_body: Bytes) -> Result<WithStatus<Json>, Rejection> {
    // let program_instance = match serde_json::from_slice::<ProgramInstance>(&body) {
    //     Ok(p) => p,
    //     Err(e) => return invalid_request(e.to_string()).await,
    // };

    // // TODO: Implement Tacosched evaluation logic here
    // match execute_tacosched_evaluation(&program_instance, &config_dir).await {
    //     Ok(result) => {
    //         let code = StatusCode::OK;
    //         Ok(warp::reply::with_status(warp::reply::json(&result), code))
    //     }
    //     Err(e) => invalid_request(e.to_string()).await,
    // }

    ready_ok()
}
