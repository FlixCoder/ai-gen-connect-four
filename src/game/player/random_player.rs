//! Player playing random actions
#![allow(dead_code)]

extern crate rand;

use self::rand::{Rng, ThreadRng};
use super::Player;
use super::super::field::Field;


pub struct PlayerRandom
{
	initialized: bool,
	pid: i32, //player ID
	rng: Box<ThreadRng>,
}

impl PlayerRandom
{
	pub fn new() -> Box<PlayerRandom>
	{
		Box::new(PlayerRandom { initialized: false, pid: 0, rng: Box::new(rand::thread_rng()) })
	}
}

impl Player for PlayerRandom
{
	#[allow(unused_variables)]
	fn init(&mut self, field:&Field, p:i32) -> bool
	{
		self.initialized = true;
		self.pid = p;
		true
	}
	
	#[allow(unused_variables)]
	fn startp(&mut self, p:i32)
	{
		//nothing
	}
	
	fn play(&mut self, field:&mut Field) -> bool
	{
		if !self.initialized { return false; }
		
		let w = field.get_w();
		let mut random = (*self.rng).gen::<u32>() % w;
		while !field.play(self.pid, random)
		{
			random = (*self.rng).gen::<u32>() % w;
		}
		
		true
	}
	
	#[allow(unused_variables)]
	fn outcome(&mut self, field:&mut Field, state:i32)
	{
		//nothing
	}
}

impl Drop for PlayerRandom
{
	fn drop(&mut self)
	{
		//nothing to do
	}
}
