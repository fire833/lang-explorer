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

use std::convert::Infallible;

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use utoipa::{OpenApi, ToSchema};
use warp::{
    reject::Rejection,
    reply::{Json, WithStatus},
    Filter,
};

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ErrorMessage {
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

pub fn health_handler() -> impl Filter<Extract = (WithStatus<Json>,), Error = Rejection> + Clone {
    warp::get()
        .and(warp::path!("readyz"))
        .or(warp::path!("livez"))
        .and(warp::path::end())
        .and_then(|_| async { ready_ok() })
}

pub fn catchall_handler() -> impl Filter<Extract = (WithStatus<Json>,), Error = Rejection> + Clone {
    warp::any().and_then(|| async { not_found() })
}

pub fn openapi_handler<O: OpenApiExtensions>(
) -> impl Filter<Extract = (WithStatus<Json>,), Error = Rejection> + Clone {
    warp::get()
        .and(warp::path!("swagger.json"))
        .or(warp::path!("openapi.json"))
        .or(warp::path!("api-docs"))
        .and(warp::path::end())
        .and_then(|_| async { O::api_docs_reply() })
}

pub fn not_found() -> Result<WithStatus<Json>, Rejection> {
    let code = StatusCode::NOT_FOUND;
    Ok(warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new(code.into(), "resource not found")),
        code,
    ))
}

pub fn ready_ok() -> Result<WithStatus<Json>, Rejection> {
    let code = StatusCode::OK;
    Ok(warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new(code.into(), "application is ready")),
        code,
    ))
}

#[allow(unused)]
pub fn invalid_authorization() -> Result<WithStatus<Json>, Rejection> {
    let code = StatusCode::UNAUTHORIZED;
    Ok(warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new(
            code.into(),
            "invalid authorization credentials provided",
        )),
        code,
    ))
}

#[allow(unused)]
pub fn invalid_request_rejection(rej: Rejection) -> Result<WithStatus<Json>, Infallible> {
    let code = StatusCode::BAD_REQUEST;
    println!("invalid request made: {rej:?}");
    Ok(warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new_from_string(
            code.into(),
            format!("invalid request: {rej:?}"),
        )),
        code,
    ))
}

pub fn invalid_request(err: String) -> Result<WithStatus<Json>, Rejection> {
    let code = StatusCode::BAD_REQUEST;
    println!("invalid request made: {err}");
    Ok(warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new_from_string(
            code.into(),
            format!("invalid request: {err}"),
        )),
        code,
    ))
}

#[allow(unused)]
pub fn internal_error(err: String) -> Result<WithStatus<Json>, Rejection> {
    let code = StatusCode::INTERNAL_SERVER_ERROR;
    Ok(warp::reply::with_status(
        warp::reply::json(&ErrorMessage::new_from_string(
            code.into(),
            format!("unable to execute request: {err}"),
        )),
        code,
    ))
}

pub trait OpenApiExtensions: OpenApi {
    fn api_docs_reply() -> Result<WithStatus<Json>, Rejection> {
        Ok(warp::reply::with_status(
            warp::reply::json(&Self::openapi()),
            StatusCode::OK,
        ))
    }

    #[allow(unused)]
    fn api_docs_print() {
        println!(
            "{}",
            Self::openapi().to_pretty_json().unwrap_or("".to_string())
        );
    }
}
