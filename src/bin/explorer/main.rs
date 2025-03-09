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

mod cli;

use core::str;

use clap::Parser;
use lang_explorer::{
    errors::LangExplorerError,
    expanders::mc::MonteCarloExpander,
    grammar::BinarySerialize,
    languages::{strings::StringValue, taco_schedule::TacoScheduleLanguage, GrammarBuilder},
};

fn main() -> Result<(), LangExplorerError> {
    let _args = cli::LangExplorerArgs::parse();
    // let toy = ToyLanguage {}.generate_grammar().unwrap();
    let taco = TacoScheduleLanguage::new(
        vec![
            StringValue::from_static_str("i"),
            StringValue::from_static_str("j"),
            StringValue::from_static_str("k"),
        ],
        vec![StringValue::from_static_str("d")],
        vec![
            StringValue::from_static_str("1"),
            StringValue::from_static_str("2"),
            StringValue::from_static_str("5"),
            StringValue::from_static_str("10"),
            StringValue::from_static_str("15"),
        ],
        vec![
            StringValue::from_static_str("1"),
            StringValue::from_static_str("2"),
            StringValue::from_static_str("5"),
            StringValue::from_static_str("10"),
            StringValue::from_static_str("15"),
        ],
        vec![
            StringValue::from_static_str("1"),
            StringValue::from_static_str("2"),
            StringValue::from_static_str("5"),
            StringValue::from_static_str("10"),
            StringValue::from_static_str("15"),
        ],
    );

    let mut mc = MonteCarloExpander::new();

    let g = taco.generate_grammar().unwrap();
    println!("Taco Schedule Grammar: {:?}", g);

    for _ in 1..10 {
        match g.generate_program_instance(&mut mc) {
            Ok(p) => {
                let data = p.serialize();
                let s = str::from_utf8(data.as_slice()).unwrap();
                println!("{s}");
            }
            Err(e) => println!("{}", e),
        }
    }

    // return args.entry();
    return Ok(());
}
