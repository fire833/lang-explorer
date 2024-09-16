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

use std::{
    fmt::{Debug, Display},
    io,
};

pub enum LangExplorerError {
    General(String),
    IOError(io::Error),
}

impl Display for LangExplorerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::General(e) => write!(f, "{}", e),
            Self::IOError(e) => write!(f, "io: {}", e),
        }
    }
}

impl Debug for LangExplorerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl From<&str> for LangExplorerError {
    fn from(value: &str) -> Self {
        Self::General(value.to_string())
    }
}

impl From<String> for LangExplorerError {
    fn from(value: String) -> Self {
        Self::General(value)
    }
}

impl From<io::Error> for LangExplorerError {
    fn from(value: io::Error) -> Self {
        Self::IOError(value)
    }
}
