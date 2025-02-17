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
    languages::{
        strings::StringValue, taco_schedule::TacoScheduleLanguage, toy_language::ToyLanguage,
        GrammarBuilder,
    },
};

fn main() -> Result<(), LangExplorerError> {
    let args = cli::LangExplorerArgs::parse();
    let toy = ToyLanguage {}.generate_grammar().unwrap();
    // let taco = TacoScheduleLanguage::new(vec![
    //     StringValue::from_static_str("i"),
    //     StringValue::from_static_str("j"),
    //     StringValue::from_static_str("k"),
    // ]);

    let mut mc = MonteCarloExpander::new();

    // let g = taco.generate_grammar().unwrap();
    // println!("Taco Schedule Grammar: {:?}", g);

    for _ in 1..5000 {
        match toy.generate_program(&mut mc) {
            Ok(p) => {
                let s = str::from_utf8(p.as_slice()).unwrap();
                println!("{s}");
            }
            Err(_) => todo!(),
        }
    }

    // return args.entry();
    return Ok(());
}
