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

use crate::{
    errors::LangExplorerError,
    grammar::Grammar,
    languages::{strings::StringValue, GrammarBuilder},
};

pub struct SpiceLanguage;

#[derive(Default, Debug)]
pub struct SpiceLanguageParams {}

impl GrammarBuilder for SpiceLanguage {
    type Term = StringValue;
    type NTerm = StringValue;
    type Params = SpiceLanguageParams;

    fn generate_grammar(
        &self,
        _params: Self::Params,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError> {
        todo!()
    }
}
