/*
*	Copyright (C) 2026 Kendall Tauser
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

use std::collections::HashSet;

use async_trait::async_trait;

use crate::{errors::LangExplorerError, evaluators::Evaluator};

enum Direction {
    Up,
    Down,
    Left,
    Right,
}

pub(crate) enum KarelInstruction {
    Move,
    TurnLeft,
    TurnRight,
    PickMarker,
    PutMarker,
    If(KarelCondition),
    IfElse(KarelCondition),
    While(KarelCondition),
}

pub(crate) enum KarelCondition {
    FrontIsClear,
    LeftIsClear,
    RightIsClear,
    MarkersPresent,
    NoMarkersPresent,
    Not(Box<KarelCondition>),
}

pub struct KarelLanguageEvaluator {
    flags: HashSet<(u32, u32)>,
    agent_direction: Direction,
    agent_position: (u32, u32),
    flag_count: u8,
}

impl KarelLanguageEvaluator {
    pub fn new() -> Self {
        Self {
            flags: HashSet::new(),
            agent_direction: Direction::Up,
            agent_position: (0, 0),
            flag_count: 0,
        }
    }

    pub fn execute(&mut self, program: Vec<KarelInstruction>) {
        for instr in program.iter() {
            match instr {
                KarelInstruction::Move => match self.agent_direction {
                    Direction::Up => {
                        self.agent_position.1 += 1;
                    }
                    Direction::Down => {
                        self.agent_position.1 -= 1;
                    }
                    Direction::Left => {
                        self.agent_position.0 -= 1;
                    }
                    Direction::Right => {
                        self.agent_position.0 += 1;
                    }
                },
                KarelInstruction::TurnLeft => match self.agent_direction {
                    Direction::Up => {
                        self.agent_direction = Direction::Left;
                    }
                    Direction::Down => {
                        self.agent_direction = Direction::Right;
                    }
                    Direction::Left => {
                        self.agent_direction = Direction::Down;
                    }
                    Direction::Right => {
                        self.agent_direction = Direction::Up;
                    }
                },
                KarelInstruction::TurnRight => match self.agent_direction {
                    Direction::Up => {
                        self.agent_direction = Direction::Right;
                    }
                    Direction::Down => {
                        self.agent_direction = Direction::Left;
                    }
                    Direction::Left => {
                        self.agent_direction = Direction::Up;
                    }
                    Direction::Right => {
                        self.agent_direction = Direction::Down;
                    }
                },
                KarelInstruction::PickMarker => {
                    let picked_up = self.flags.remove(&self.agent_position);
                    if picked_up {
                        self.flag_count += 1;
                    }
                }
                KarelInstruction::PutMarker => {
                    if self.flag_count > 0 {
                        self.flags.insert(self.agent_position);
                        self.flag_count -= 1;
                    }
                }
                KarelInstruction::If(karel_condition) => todo!(),
                KarelInstruction::IfElse(karel_condition) => todo!(),
                KarelInstruction::While(karel_condition) => todo!(),
            }
        }
    }
}

#[async_trait]
impl Evaluator for KarelLanguageEvaluator {
    type Metric = u64;

    async fn evaluate(&self, program: Vec<u8>) -> Result<Self::Metric, LangExplorerError> {
        todo!()
    }
}
