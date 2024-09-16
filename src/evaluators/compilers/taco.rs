/*
*	Copyright (C) 2024 Kendall Tauser
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
    fn to_argument_directive(&self) -> &'static str {
        match self {
            TacoCompileMode::CompileComputeKernel => "-print-compute",
            TacoCompileMode::CompileAssemblyKernel => "-print-assembly",
            TacoCompileMode::CompileEvaluateKernel => "-print-evalute",
            TacoCompileMode::CompileLibrary => "-print-kernels",
        }
    }
}

pub struct TacoCompiler {}

impl TacoCompiler {
    /// Takes a TACO expression and a schedule, and compiles the output.
    async fn compile(
        &self,
        executable: String,
        input_expression: String,
        tensor_formats: HashMap<String, String>,
        mode: TacoCompileMode,
        threads: u32,
        schedule: String,
    ) -> Result<Vec<u8>, LangExplorerError> {
        let mut fargs = vec![];
        for (tensor, format) in tensor_formats {
            fargs.push(format!("-f={}:{}", tensor, format));
        }

        match Command::new(executable)
            .arg(input_expression)
            .arg(format!("-s=\"{}\"", schedule))
            .arg(mode.to_argument_directive())
            .args(fargs)
            .arg(format!("-nthreads={}", threads))
            .output()
        {
            Ok(o) => Ok(o.stdout),
            Err(e) => Err(e.into()),
        }
    }
}
