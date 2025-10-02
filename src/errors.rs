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
    fmt::{Debug, Display},
    io,
    string::FromUtf8Error,
};

use burn::{record::RecorderError, tensor::DataError};
use tokio::sync::mpsc::error::SendError;

use crate::languages::ProgramResult;

pub enum LangExplorerError {
    General(String),
    IOError(io::Error),
    CCError(cc::Error),
    FromUtf8Error(FromUtf8Error),
    RecorderError(RecorderError),
    SendError(String),
    DataError(DataError),
    CSVError(csv::Error),
    ReqwestError(reqwest::Error),
    SerdeJSONError(serde_json::Error),
}

impl Display for LangExplorerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::General(e) => write!(f, "{}", e),
            Self::IOError(e) => write!(f, "io: {}", e),
            Self::CCError(e) => write!(f, "cc: {}", e),
            Self::FromUtf8Error(e) => write!(f, "utf8: {}", e),
            Self::RecorderError(e) => write!(f, "recorder: {}", e),
            Self::SendError(e) => write!(f, "sender: {}", e),
            Self::DataError(e) => write!(f, "data: {:?}", e),
            Self::CSVError(e) => write!(f, "csv: {}", e),
            Self::ReqwestError(e) => write!(f, "req: {}", e),
            Self::SerdeJSONError(e) => write!(f, "json: {}", e),
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

impl From<cc::Error> for LangExplorerError {
    fn from(value: cc::Error) -> Self {
        Self::CCError(value)
    }
}

impl From<FromUtf8Error> for LangExplorerError {
    fn from(value: FromUtf8Error) -> Self {
        Self::FromUtf8Error(value)
    }
}

impl From<RecorderError> for LangExplorerError {
    fn from(value: RecorderError) -> Self {
        Self::RecorderError(value)
    }
}

impl From<SendError<ProgramResult>> for LangExplorerError {
    fn from(value: SendError<ProgramResult>) -> Self {
        Self::SendError(value.to_string())
    }
}

impl From<DataError> for LangExplorerError {
    fn from(value: DataError) -> Self {
        Self::DataError(value)
    }
}

impl From<csv::Error> for LangExplorerError {
    fn from(value: csv::Error) -> Self {
        Self::CSVError(value)
    }
}

impl From<reqwest::Error> for LangExplorerError {
    fn from(value: reqwest::Error) -> Self {
        Self::ReqwestError(value)
    }
}

impl From<serde_json::Error> for LangExplorerError {
    fn from(value: serde_json::Error) -> Self {
        Self::SerdeJSONError(value)
    }
}
