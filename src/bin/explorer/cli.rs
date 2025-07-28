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

use lang_explorer::{
    errors::LangExplorerError, expanders::ExpanderWrapper, languages::LanguageWrapper,
};

use crate::api;

#[derive(clap::Parser)]
pub(super) struct LangExplorerArgs {
    #[command(subcommand)]
    cmd: Option<Subcommand>,
}

impl LangExplorerArgs {
    pub(super) async fn entry(&self) -> Result<(), LangExplorerError> {
        match &self.cmd {
            Some(cmd) => match cmd {
                Subcommand::Explore => todo!(),
                Subcommand::MPIExplore => todo!(),
                Subcommand::Generate {
                    language: _,
                    expander: _,
                    count: _,
                } => Ok(()),
                Subcommand::Serve { address, port } => {
                    Ok(api::start_server(address.as_str(), *port).await)
                }
            },
            None => return Err("no command provided".into()),
        }
    }
}

#[derive(clap::Subcommand)]
enum Subcommand {
    /// Run lang-explorer to explore a problem space.
    #[command()]
    Explore,

    /// Run lang-explorer in an MPI environment.
    #[command()]
    MPIExplore,

    /// Generate a new program in a given language from
    /// a given specification with a given expander.
    #[command()]
    Generate {
        #[arg(short, long, value_enum)]
        language: LanguageWrapper,

        #[arg(short, long, value_enum)]
        expander: ExpanderWrapper,

        #[arg(short, long, default_value_t = 1)]
        count: u64,
    },

    /// Run an API server to handle requests for
    /// generated programs.
    #[command()]
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
