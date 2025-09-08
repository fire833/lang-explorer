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

use burn::backend::{Cuda, NdArray};
use lang_explorer::{
    errors::LangExplorerError,
    expanders::ExpanderWrapper,
    languages::{GenerateParams, LanguageWrapper},
};

use crate::api;

#[derive(clap::Parser)]
pub(super) struct LangExplorerArgs {
    /// Specify the location model files are stored.
    #[arg(short, long, default_value_t = String::from("./models"))]
    model_dir: String,

    /// Specify the Ollama hostname for calling ollama crap.
    #[arg(short, long, default_value_t = String::from("http://localhost:11434"))]
    ollama_host: String,

    /// Specify the location for storing outputs.
    #[arg(short, long, default_value_t = String::from("./lang-explorer-python/results"))]
    output_dir: String,

    /// Specify the subcommand to be run.
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
                Subcommand::EmbedGen {
                    language: _,
                    expander: _,
                    count: _,
                } => Ok(()),
                Subcommand::Serve {
                    address,
                    port,
                    cuda,
                } => {
                    if *cuda {
                        let _ = api::start_server::<Cuda>(
                            address.as_str(),
                            *port,
                            self.model_dir.clone(),
                            self.ollama_host.clone(),
                            self.output_dir.clone(),
                        )
                        .await;
                    } else {
                        let _ = api::start_server::<NdArray>(
                            address.as_str(),
                            *port,
                            self.model_dir.clone(),
                            self.ollama_host.clone(),
                            self.output_dir.clone(),
                        )
                        .await;
                    }
                    Ok(())
                }
            },
            None => Err("no command provided".into()),
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

    /// Generate a new set of programs along with
    /// corresponding embeddings.
    EmbedGen {
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

        /// Toggle CUDA backend.
        #[arg(short, long, default_value_t = false)]
        cuda: bool,
    },
}

fn default_bind_addr() -> String {
    "0.0.0.0".into()
}

fn default_port() -> u16 {
    8080
}
