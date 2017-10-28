//! IO player (human interface player))
#![allow(dead_code)]

use std::io;
use std::io::Write;
use super::Player;
use super::super::field::Field;


pub struct PlayerIO
{
	initialized: bool,
	pid: i32, //player ID
}

impl PlayerIO
{
	pub fn new() -> Box<PlayerIO>
	{
		Box::new(PlayerIO { initialized: false, pid: 0 })
	}
}

impl Player for PlayerIO
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
		
		field.print();
		println!("");
		
		#[allow(unused_assignments)]
		let mut x:u32 = 0;
		loop
		{
			print!("Enter column (starting at 0): ");
			io::stdout().flush().expect("Failed flushing stdout!");
			
			let mut str = String::new();
			match io::stdin().read_line(&mut str)
			{
				Err(_) =>
				{
					println!("Failed reading input line! Try again!");
					continue;
				},
				_ => {},
			}
			match str.trim().parse()
			{
				Ok(num) =>
				{
					x = num;
					if !field.is_valid_play(x)
					{
						println!("No possible move! Try again!");
						continue;
					}
					break;
				},
				_ =>
				{
					println!("Input not valid, try again!");
					continue;
				},
			}
		}
		println!("");
		
		field.play(self.pid, x)
	}
	
	#[allow(unused_variables)]
	fn outcome(&mut self, field:&mut Field, state:i32)
	{
		field.print();
		println!("");
		if state == -1 { println!("Draw!"); }
		else { println!("Player {} won the game!", if state == 1 { "X" } else { "O" }); }
	}
}

impl Drop for PlayerIO
{
	fn drop(&mut self)
	{
		//nothing to do
	}
}
