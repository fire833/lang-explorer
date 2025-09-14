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

use lang_explorer::errors::LangExplorerError;

use crate::api::start_server;

#[derive(clap::Parser)]
pub(crate) struct EvaluatorArgs {
    #[command(subcommand)]
    cmd: Option<Subcommand>,
}

impl EvaluatorArgs {
    pub async fn entry(&self) -> Result<(), LangExplorerError> {
        match &self.cmd {
            Some(cmd) => match cmd {
                Subcommand::Serve { address, port } => Ok(start_server(&address, *port).await),
            },
            None => Ok(()),
        }
    }
}

#[derive(clap::Subcommand)]
enum Subcommand {
    Serve {
        /// Specify the address to bind to.
        #[arg(short, long, default_value_t = default_bind_addr())]
        address: String,

        /// Specify the port to listen on for the server.
        #[arg(short, long, default_value_t = default_port())]
        port: u16,
    },
}

fn default_bind_addr() -> String {
    "0.0.0.0".into()
}

fn default_port() -> u16 {
    8080
}
