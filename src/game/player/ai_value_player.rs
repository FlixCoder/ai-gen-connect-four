//! trained NN to represent value heuristic and use minimax
#![allow(dead_code)]

extern crate ernn;

use self::ernn::{NN, Activation};
use super::Player;
use super::super::field::Field;
use std::f64;

const DEEPNESS:u32 = 3; //default recursion limit

//values for a won or lost game in minimax and heuristic (neural net outputs should be a lot closer to zero)
const VAL_MAX:f64 = 10002.0; //f64::MAX
const VAL_MIN:f64 = -10002.0; //f64::MIN


pub struct PlayerAIValue
{
	initialized: bool,
	pid: i32, //player ID
	startp: i32, //starting player
	deepness: u32, //minimax search depth
	nn: Option<NN>, //neural network for neutral state evaluation (value based on starting player)
}

impl PlayerAIValue
{
	///creates a new AI Player, which uses a value NN, uses default minimax depth
	///net = neural net to evualuate positions
	pub fn new(net:NN) -> Box<PlayerAIValue>
	{
		PlayerAIValue::new_deep(net, DEEPNESS)
	}

	///creates a new AI Player, which uses a value NN
	///net = neural net to evualuate positions
	///deep = minimax deepness: pass 0 to use default deepness
	pub fn new_deep(net:NN, mut deep:u32) -> Box<PlayerAIValue>
	{
		if deep < 1 { deep = DEEPNESS; } //change invalid depth into default
		Box::new(PlayerAIValue { initialized: false, pid: 0, startp: 0, deepness: deep, nn: Some(net) })
	}
	
	//raw field
	fn field_to_input(field:&mut Field, p:i32) -> Vec<f64>
	{ //input: p = start player
		let op:i32 = if p == 1 { 2 } else { 1 }; //other player
		let mut input:Vec<f64> = Vec::with_capacity(field.get_size() as usize);
		//1 nodes for every square: -1 enemy, 0 free, 1 own
		for val in field.get_field().iter()
		{
			if *val == p { input.push(1f64); }
			else if *val == op { input.push(-1f64); }
			else { input.push(0f64); } //empty square
		}
		//return
		input
	}
	
	//returns value of board position: +1.0 player wins, -1.0 other player wins, 0.0 draw or even board
	fn heur(&self, field:&mut Field, p:i32, deep:u32) -> f64 //p = player. translated from start player by (value * -1) if they are not same.
	{
		let op = if p == 1 {2} else {1};
		let state = field.get_state(); //return best or worst value on win/loose (neutral on tie)
		if state == -1 { return 0.0; }
		else if state == p { return VAL_MAX - deep as f64; }
		else if state == op { return VAL_MIN + deep as f64; }
		else
		{ //game running -> evaluate
			let nn = self.nn.as_ref().unwrap();
			let state = PlayerAIValue::field_to_input(field, self.startp);
			let factor = if p == self.startp { 1.0 } else { -1.0 }; //if p is not startp value has to be reversed
			
			let result = nn.run(&state);
			let value = factor * result[0];
			
			return value;
		}
	}
	
	fn minimax(&self, field:&mut Field, p:i32, deep:u32) -> f64
	{
		let op = if p == 1 {2} else {1};
		if deep > self.deepness { return self.heur(field, if deep%2 == 0 {op} else {p}, deep); } //leaf node -> return evaluated heuristic, mechanism to get heur always for same player
		let state = field.get_state(); //return early on game end
		if state == -1 { return 0.0; }
		else if state == p { return if deep%2 == 0 {VAL_MIN + deep as f64} else {VAL_MAX - deep as f64}; }
		else if state == op { return if deep%2 == 0 {VAL_MAX - deep as f64} else {VAL_MIN + deep as f64}; }
		
		//else: game running -> go deeper
		let mut heur = if deep%2 == 0 { f64::INFINITY } else { f64::NEG_INFINITY };
		for i in 0..field.get_w()
		{
			if field.play(p, i)
			{
				let val = self.minimax(field, op, deep+1);
				field.undo();
				if (deep%2 == 0 && val < heur) || (deep%2 == 1 && val > heur) //min or max according to which player's turn it is
				{
					heur = val;
				}
			}
		}
		heur
	}
}

impl Player for PlayerAIValue
{
	///initializes the player with game information
	///field = field object to play on
	///p = player id (1 or 2(?))
	#[allow(unused_variables)]
	fn init(&mut self, field:&Field, p:i32) -> bool
	{
		if self.deepness < 1 { return false; } //invalid player, could cause bugs else
		
		self.pid = p;
		
		let nn = self.nn.as_ref().unwrap();
		if nn.get_outputs() == 1 && nn.get_inputs() == field.get_w() * field.get_h() && nn.get_out_act() == Activation::Tanh
		{
			self.initialized = true;
		}
		else
		{
			self.initialized = false;
			eprintln!("Warning: PlayerAIValue not initialized, because NN does not fit the requirements!");
		}
		
		self.initialized
	}
	
	#[allow(unused_variables)]
	fn startp(&mut self, p:i32)
	{
		self.startp = p;
	}
	
	fn play(&mut self, field:&mut Field) -> bool
	{
		if !self.initialized { return false; }
		
		let p = self.pid;
		let op = if p == 1 {2} else {1};
		
		//decide which action x to take
		let mut x:u32 = 0;
		let mut max = f64::NEG_INFINITY;
		//decide by evaluation
		for i in 0..field.get_w()
		{
			let mut val = f64::NEG_INFINITY;
			if field.play(p, i)
			{
				val = self.minimax(field, op, 2);
				field.undo();
			}
			if max < val || !field.is_valid_play(x)
			{
				max = val;
				x = i;
			}
		}
		
		//debug
		//println!("Heur: {}", max);
		
		//play (actually should always be true, unless game was finished before method invocation)
		field.play(p, x)
	}
	
	#[allow(unused_variables)]
	fn outcome(&mut self, field:&mut Field, state:i32)
	{
		//not needed
	}
}

impl Drop for PlayerAIValue
{
	fn drop(&mut self)
	{
		//not needed
	}
}
