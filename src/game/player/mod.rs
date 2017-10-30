#![allow(dead_code)]

pub mod io_player;
pub mod random_player;
pub mod minimax_player;
pub mod ai_value_player;

use super::field::Field;


pub trait Player:Drop
{
	fn init(&mut self, field:&Field, p_id:i32) -> bool;
	fn startp(&mut self, p_id:i32);
	fn play(&mut self, field:&mut Field) -> bool;
	fn outcome(&mut self, field:&mut Field, state:i32);
}
