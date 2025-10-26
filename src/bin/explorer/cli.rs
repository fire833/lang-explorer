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
    experiments::generate::{GenerateInput, GenerateOutput},
    languages::LanguageWrapper,
};

use crate::api;

#[derive(clap::Parser)]
pub(super) struct LangExplorerArgs {
    /// Specify the location model files are stored.
    #[arg(short, long, default_value_t = String::from("./models"))]
    models_dir: String,

    /// Specify the Ollama hostname for calling ollama stuff.
    #[arg(short, long, default_value_t = String::from("https://ollama.soonerhpclab.org"))]
    ollama_host: String,

    /// Specify the Doc2Vec server hostname for calling d2v stuff.
    #[arg(short, long, default_value_t = String::from("https://d2v.soonerhpclab.org"))]
    d2v_host: String,

    /// Specify the location for storing outputs.
    #[arg(long, default_value_t = String::from("./lang-explorer-python/results"))]
    output_dir: String,

    /// Toggle CUDA backend.
    #[arg(short, long, default_value_t = false)]
    cuda: bool,

    /// Specify the subcommand to be run.
    #[command(subcommand)]
    cmd: Option<Subcommand>,
}

impl LangExplorerArgs {
    pub(super) async fn entry(&self) -> Result<(), LangExplorerError> {
        match &self.cmd {
            Some(cmd) => match cmd {
                Subcommand::DefaultConfig => {
                    let default_config = GenerateInput::default();
                    let config_str = serde_json::to_string_pretty(&default_config)?;
                    println!("{}", config_str);
                    Ok(())
                }
                Subcommand::Generate {
                    language,
                    expander,
                    redo,
                    config,
                } => {
                    let config = match (redo, config) {
                        (None, None) => {
                            println!("using defaults");
                            GenerateInput::default()
                        }
                        (None, Some(file)) => GenerateInput::from_file(file.as_str())?,
                        (Some(idx), _) => {
                            GenerateInput::from_experiment_id(&self.output_dir, language, *idx)?
                        }
                    };

                    let res = if self.cuda {
                        config
                            .execute::<Cuda>(
                                language.clone(),
                                expander.clone(),
                                self.models_dir.clone(),
                                self.ollama_host.clone(),
                                self.d2v_host.clone(),
                                self.output_dir.clone(),
                            )
                            .await?
                    } else {
                        config
                            .execute::<NdArray>(
                                language.clone(),
                                expander.clone(),
                                self.models_dir.clone(),
                                self.ollama_host.clone(),
                                self.d2v_host.clone(),
                                self.output_dir.clone(),
                            )
                            .await?
                    };

                    res.write(self.output_dir.clone(), None)?;

                    Ok(())
                }
                Subcommand::Serve { address, port } => {
                    if self.cuda {
                        let _ = api::start_server::<Cuda>(
                            address.as_str(),
                            *port,
                            self.models_dir.clone(),
                            self.ollama_host.clone(),
                            self.d2v_host.clone(),
                            self.output_dir.clone(),
                        )
                        .await;
                    } else {
                        let _ = api::start_server::<NdArray>(
                            address.as_str(),
                            *port,
                            self.models_dir.clone(),
                            self.ollama_host.clone(),
                            self.d2v_host.clone(),
                            self.output_dir.clone(),
                        )
                        .await;
                    }
                    Ok(())
                }
                Subcommand::RedoExperiments {
                    language,
                    redo,
                    write_to_new,
                } => {
                    let input =
                        GenerateInput::from_experiment_id(&self.output_dir, language, *redo)?;
                    let mut res =
                        GenerateOutput::from_experiment_id(&self.output_dir, language, *redo)
                            .await?;

                    res.do_experiments(&input)?;
                    if *write_to_new {
                        res.write(&self.output_dir, None)?;
                    } else {
                        res.write(&self.output_dir, Some(*redo))?;
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
    /// Generate a new set of programs, optionally along with
    /// corresponding embeddings.
    Generate {
        /// Specify the language to generate programs in.
        #[arg(short, long, value_enum)]
        language: LanguageWrapper,

        /// Specify the expander to use.
        #[arg(short, long, value_enum)]
        expander: ExpanderWrapper,

        /// Optionally redo another experiment by ID.
        #[arg(short, long)]
        redo: Option<usize>,

        /// Optionally specify a configuration file.
        #[arg(short, long)]
        config: Option<String>,
    },

    /// Generate a default configuration file.
    DefaultConfig,

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

    RedoExperiments {
        /// Specify the language to redo experiments for.
        #[arg(short, long, value_enum)]
        language: LanguageWrapper,

        /// Specify the experiment number to load data in and recompute.
        #[arg(short, long)]
        redo: usize,

        /// Toggle writing to a new experiment ID instead of overwriting
        /// the current experiment results.
        #[arg(short, long)]
        write_to_new: bool,
    },
}

fn default_bind_addr() -> String {
    "0.0.0.0".into()
}

fn default_port() -> u16 {
    8080
}
