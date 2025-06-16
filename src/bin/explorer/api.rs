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

use warp::{reply::reply, Filter};

use crate::glue::GenerateParams;

pub async fn start_server(addr: &str, port: u16) {
    let get_local_vpn = warp::post()
        .and(warp::path!("v1" / "generate"))
        .and(warp::path::end())
        .and(warp::body::json::<GenerateParams>())
        .and_then(generate);

    let routes = get_local_vpn;

    warp::serve(routes)
        .run(SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::from_str(addr).expect("invalid bind ip address"),
            port,
        )))
        .await;
}

async fn generate(_params: GenerateParams) -> Result<impl warp::Reply, warp::Rejection> {
    Ok(reply())
}
