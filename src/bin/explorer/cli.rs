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

#[derive(clap::Parser)]
pub struct LangExplorerArgs {
    #[command(subcommand)]
    cmd: Option<Subcommand>,
}

impl LangExplorerArgs {
    pub fn entry(&self) -> Result<(), LangExplorerError> {
        match &self.cmd {
            Some(cmd) => match cmd {
                Subcommand::Explore => todo!(),
                Subcommand::MPIExplore => todo!(),
                Subcommand::Generate => todo!(),
                Subcommand::Serve => todo!(),
            },
            None => return Err("no command provided".into()),
        }
    }
}

#[derive(clap::Subcommand)]
pub enum Subcommand {
    /// Run lang-explorer to explore a problem space.
    #[command()]
    Explore,

    /// Run lang-explorer in an MPI environment.
    #[command()]
    MPIExplore,

    /// Generate a new program in a given language from
    /// a given specification with a given expander.
    #[command()]
    Generate,

    /// Run an API server to handle requests for
    /// generated programs.
    #[command()]
    Serve,
}
