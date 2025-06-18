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

use serde::{Deserialize, Serialize};

use crate::{
    errors::LangExplorerError,
    grammar::{elem::GrammarElement, Grammar},
};

use super::{
    strings::{nterminal_str, terminal_str, StringValue},
    GrammarBuilder,
};

nterminal_str!(NT_TABLE, "nt_table");
nterminal_str!(NT_CHAIN, "nt_chain");
nterminal_str!(NT_RULE, "nt_rule");

terminal_str!(TABLE, "table");
terminal_str!(CHAIN, "chain");
terminal_str!(ADDRESS_FAMILY_INET, "inet");
terminal_str!(ADDRESS_FAMILY_V4, "ip");
terminal_str!(ADDRESS_FAMILY_V6, "ip6");
terminal_str!(ADDRESS_FAMILY_ARP, "arp");
terminal_str!(ADDRESS_FAMILY_BRIDGE, "bridge");
terminal_str!(ADDRESS_FAMILY_NETDEV, "netdev");

pub struct NFTRulesetLanguage;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NFTRulesetParams {}

impl GrammarBuilder for NFTRulesetLanguage {
    type Term = StringValue;
    type NTerm = StringValue;
    type Params<'de> = NFTRulesetParams;

    fn generate_grammar<'de>(
        _params: Self::Params<'de>,
    ) -> Result<Grammar<Self::Term, Self::NTerm>, LangExplorerError> {
        todo!()
    }
}
