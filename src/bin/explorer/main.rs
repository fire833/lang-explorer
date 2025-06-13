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

mod api;
mod cli;
mod generate;

use clap::Parser;
use lang_explorer::errors::LangExplorerError;

#[tokio::main]
async fn main() -> Result<(), LangExplorerError> {
    let args = cli::LangExplorerArgs::parse();
    return args.entry().await;

    // // let toy = ToyLanguage {}.generate_grammar().unwrap();
    // let taco = TacoScheduleLanguage::new(
    //     vec![
    //         StringValue::from_static_str("i"),
    //         StringValue::from_static_str("j"),
    //         StringValue::from_static_str("k"),
    //     ],
    //     vec![StringValue::from_static_str("d")],
    //     vec![StringValue::from_static_str("f")],
    //     vec![
    //         StringValue::from_static_str("1"),
    //         StringValue::from_static_str("2"),
    //         StringValue::from_static_str("5"),
    //         StringValue::from_static_str("10"),
    //         StringValue::from_static_str("15"),
    //     ],
    //     vec![
    //         StringValue::from_static_str("1"),
    //         StringValue::from_static_str("2"),
    //         StringValue::from_static_str("5"),
    //         StringValue::from_static_str("10"),
    //         StringValue::from_static_str("15"),
    //     ],
    //     vec![
    //         StringValue::from_static_str("1"),
    //         StringValue::from_static_str("2"),
    //         StringValue::from_static_str("5"),
    //         StringValue::from_static_str("10"),
    //         StringValue::from_static_str("15"),
    //     ],
    // );

    // let mut mc = MonteCarloExpander::new();

    // let g = taco.generate_grammar().unwrap();
    // println!("Taco Schedule Grammar: {:?}", g);

    // for i in 1..15 {
    //     match g.generate_program_instance(&mut mc) {
    //         Ok(p) => {
    //             println!(
    //                 "instance {}: {}",
    //                 i,
    //                 str::from_utf8(p.serialize().as_slice()).unwrap()
    //             );
    //         }
    //         Err(e) => println!("{}", e),
    //     }
    // }

    // return args.entry();
}
