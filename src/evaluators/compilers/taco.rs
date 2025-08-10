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

use std::{collections::HashMap, process::Command};

use crate::errors::LangExplorerError;

/// Enumeration of supported compiler modes for taco.
/// When you select one of these modes, it will affect the
/// arguments that are passed to the taco process.
#[allow(unused)]
pub enum TacoCompileMode {
    /// Print the compute kernel (default).
    CompileComputeKernel,

    /// Print the assembly kernel.
    CompileAssemblyKernel,

    /// Print the evaluate kernel.
    CompileEvaluateKernel,

    /// Print all kernels as a C library.
    CompileLibrary,
}

impl TacoCompileMode {
    #[allow(unused)]
    fn to_argument_directive(&self, file: String) -> String {
        match self {
            TacoCompileMode::CompileComputeKernel => format!("-write-compute={}", file),
            TacoCompileMode::CompileAssemblyKernel => format!("-write-assembly={}", file),
            TacoCompileMode::CompileEvaluateKernel => format!("-write-evalute={}", file),
            TacoCompileMode::CompileLibrary => format!("-write-source={}", file),
        }
    }
}

/// A wrapper for doing taco compilation stuff.
#[allow(unused)]
pub struct TacoCompiler {}

impl TacoCompiler {
    /// Takes a TACO expression and a schedule along with some other arguments,
    /// and compiles the output to a file.
    #[allow(unused)]
    async fn compile(
        &self,
        executable: String,
        outfile: String,
        input_expression: String,
        tensor_formats: HashMap<String, String>,
        mode: TacoCompileMode,
        threads: u32,
        schedule: String,
    ) -> Result<(), LangExplorerError> {
        let mut fargs = vec![];
        for (tensor, format) in tensor_formats {
            fargs.push(format!("-f={}:{}", tensor, format));
        }

        match Command::new(executable)
            .arg(input_expression)
            .arg(format!("-s=\"{}\"", schedule))
            .arg(mode.to_argument_directive(outfile))
            .args(fargs)
            .arg(format!("-nthreads={}", threads))
            .output()
        {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}
