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

use std::path::Path;

use crate::errors::LangExplorerError;

#[allow(unused)]
pub struct CCompiler {}

impl CCompiler {
    #[allow(unused)]
    async fn compile(infile: &Path, objectfile: &Path) -> Result<(), LangExplorerError> {
        let dir = objectfile.parent();
        let out = objectfile.file_name();

        match (out, dir) {
            (Some(f), Some(d)) => match cc::Build::new()
                .file(infile)
                .out_dir(d)
                .shared_flag(true)
                .try_compile(format!("{:?}", f).as_str())
            {
                Ok(_) => Ok(()),
                Err(e) => Err(e.into()),
            },
            (None, None) | (Some(_), None) | (None, Some(_)) => Err(LangExplorerError::General(
                "outfile must have an output file".to_string(),
            )),
        }
    }
}
