#![allow(dead_code)]

extern crate ernn;

pub mod field;
pub mod player;

use self::ernn::NN;
use self::field::Field;
use self::player::Player;
use self::player::io_player::PlayerIO;
use self::player::random_player::PlayerRandom;
use self::player::minimax_player::PlayerMinimax;
use self::player::ai_value_player::PlayerAIValue;


#[derive(Debug)]
pub enum PlayerType { None, IO, Random, Minimax, AIValue }

pub struct Game
{
	field: Field,
	p1: Option<Box<Player>>,
	p2: Option<Box<Player>>,
	startp: u32
}

impl Game
{
	pub fn new() -> Game
	{
		Game { field: Field::new(7, 6), p1: None, p2: None, startp: 1 }
	}
	
	fn map_player(p:PlayerType) -> Option<Box<Player>>
	{
		match p
		{
			PlayerType::None => None,
			PlayerType::IO => Some(PlayerIO::new()),
			PlayerType::Random => Some(PlayerRandom::new()),
			PlayerType::Minimax => Some(PlayerMinimax::new()),
			_ => None,
		}
	}
	
	fn map_player_nn(p:PlayerType, nn:NN) -> Option<Box<Player>>
	{
		match p
		{
			PlayerType::AIValue => Some(PlayerAIValue::new(nn)),
			_ => None,
		}
	}

	fn map_player_nn_deep(p:PlayerType, nn:NN, deep:u32) -> Option<Box<Player>>
	{
		match p
		{
			PlayerType::AIValue => Some(PlayerAIValue::new_deep(nn, deep)),
			_ => None,
		}
	}
	
	pub fn set_player1(&mut self, p:PlayerType) -> bool
	{
		self.p1 = Game::map_player(p);
		
		if self.p1.is_some()
		{
			if !self.p1.as_mut().unwrap().init(&self.field, 1)
			{
				self.p1 = None;
				return false;
			}
		}
		true
	}
	
	pub fn set_player2(&mut self, p:PlayerType) -> bool
	{
		self.p2 = Game::map_player(p);
		
		if self.p2.is_some()
		{
			if !self.p2.as_mut().unwrap().init(&self.field, 2)
			{
				self.p2 = None;
				return false;
			}
		}
		true
	}
	
	pub fn set_player1_nn(&mut self, p:PlayerType, nn:NN) -> bool
	{
		self.p1 = Game::map_player_nn(p, nn);
		
		if self.p1.is_some()
		{
			if !self.p1.as_mut().unwrap().init(&self.field, 1)
			{
				self.p1 = None;
				return false;
			}
		}
		true
	}
	
	pub fn set_player2_nn(&mut self, p:PlayerType, nn:NN) -> bool
	{
		self.p2 = Game::map_player_nn(p, nn);
		
		if self.p2.is_some()
		{
			if !self.p2.as_mut().unwrap().init(&self.field, 2)
			{
				self.p2 = None;
				return false;
			}
		}
		true
	}

	pub fn set_player1_nn_deep(&mut self, p:PlayerType, nn:NN, deep:u32) -> bool
	{
		self.p1 = Game::map_player_nn_deep(p, nn, deep);
		
		if self.p1.is_some()
		{
			if !self.p1.as_mut().unwrap().init(&self.field, 1)
			{
				self.p1 = None;
				return false;
			}
		}
		true
	}
	
	pub fn set_player2_nn_deep(&mut self, p:PlayerType, nn:NN, deep:u32) -> bool
	{
		self.p2 = Game::map_player_nn_deep(p, nn, deep);
		
		if self.p2.is_some()
		{
			if !self.p2.as_mut().unwrap().init(&self.field, 2)
			{
				self.p2 = None;
				return false;
			}
		}
		true
	}
	
	pub fn set_start_player(&mut self, p:u32) -> bool
	{
		if p < 1 || p > 2 { return false; }
		self.startp = p;
		true
	}
	
	pub fn is_ready(&self) -> bool
	{
		self.p1.is_some() && self.p2.is_some()
	}
	
	pub fn play(&mut self) -> bool
	{
		if !self.is_ready() { return false; }
		
		let p1 = self.p1.as_mut().unwrap();
		let p2 = self.p2.as_mut().unwrap();
		
		self.field.reset();
		let mut turn1:bool = self.startp == 1;
		let mut state = 0;
		p1.startp(self.startp as i32);
		p2.startp(self.startp as i32);
		
		while state == 0
		{
			if turn1
			{
				if !p1.play(&mut self.field)
				{ println!("Warning: player 1 did not play!"); }
			}
			else
			{
				if !p2.play(&mut self.field)
				{ println!("Warning: player 2 did not play!"); }
			}
			turn1 = !turn1;
			state = self.field.get_state();
			//self.field.print(); //debug
		}
		
		p1.outcome(&mut self.field, state);
		p2.outcome(&mut self.field, state);
		
		true
	}
	
	pub fn play_many(&mut self, num:u32, every:u32) -> (f64, f64, f64)
	{
		if num<1 { return (-1.0, -1.0, -1.0); }
		
		let mut p1win:u32 = 0;
		let mut draw:u32 = 0;
		let mut p2win:u32 = 0;
		
		for i in 0..num
		{
			if i > 0 && i%every == 0 { self.startp = if self.startp == 1 { 2 } else { 1 }; } //switch sides every "every" game
			if !self.play() { return (-1.0, -1.0, -1.0); }
			match self.field.get_state()
			{
				-1 => draw += 1,
				1 => p1win += 1,
				2 => p2win += 1,
				_ => println!("Error: game ended while running!"),
			}
		}
		
		let p1wr:f64 = (p1win as f64)/(num as f64)*100.0;
		let drawrate:f64 = (draw as f64)/(num as f64)*100.0;
		let p2wr:f64 = (p2win as f64)/(num as f64)*100.0;
		
		(p1wr, drawrate, p2wr)
	}
}
