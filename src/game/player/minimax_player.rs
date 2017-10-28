//! Minimax player
#![allow(dead_code)]

use super::Player;
use super::super::field::Field;
use std::thread;
use std::f64;

const DEEPNESS:u32 = 5; //recursion limit


pub struct PlayerMinimax
{
	initialized: bool,
	pid: i32, //player ID
}

impl PlayerMinimax
{
	pub fn new() -> Box<PlayerMinimax>
	{
		Box::new(PlayerMinimax { initialized: false, pid: 0 })
	}
	
	fn heur(field:&mut Field, p:i32, deep:u32) -> f64
	{
		let op = if p == 1 {2} else {1};
		let state = field.get_state(); //return best or worst value on win/loose (neutral on tie)
		if state == -1 { return 0.0; }
		else if state == p { return 10002.0 - deep as f64; }
		else if state == op { return -10002.0 + deep as f64; }
		else
		{ //game running -> evaluate
			//count 2 and 3 rows of player and enemy, add up free squares next to player stones
			let mut pr2 = 0;
			let mut pr3 = 0;
			let mut heur = 0.0;
			let value = 1f64; //value of stones around free space
			for y in 0..field.get_h()
			{
				for x in 0..field.get_w()
				{
					let val = field.get_val(x, y);
					if val == p
					{
						if field.get_val(x+1, y) == p { if field.get_val(x+2, y) == p { pr3 += 1; } else { pr2 += 1; } }
						if field.get_val(x+1, y+1) == p { if field.get_val(x+2, y+2) == p { pr3 += 1; } else { pr2 += 1; } }
						if field.get_val(x, y+1) == p { if field.get_val(x, y+2) == p { pr3 += 1; } else { pr2 += 1; } }
						if x>=1 && field.get_val(x-1, y+1) == p { if x>=2 && field.get_val(x-2, y+2) == p { pr3 += 1; } else { pr2 += 1; } }
						//free squares
						if y>=1
						{
							if x>=1 && field.get_val(x-1, y-1) == 0 { heur += value; }
							if field.get_val(x, y-1) == 0 { heur += value; }
							if x<field.get_w()-1 && field.get_val(x+1, y-1) == 0 { heur += value; }
						}
						if x>=1 && field.get_val(x-1, y) == 0 { heur += value; }
						if x<field.get_w()-1 && field.get_val(x+1, y) == 0 { heur += value; }
						if y<field.get_h()-1
						{
							if x>=1 && field.get_val(x-1, y+1) == 0 { heur += value; }
							if x<field.get_w()-1 && field.get_val(x+1, y+1) == 0 { heur += value; }
						}
					}
					else if val == op
					{
						if field.get_val(x+1, y) == op { if field.get_val(x+2, y) == op { pr3 -= 1; } else { pr2 -= 1; } }
						if field.get_val(x+1, y+1) == op { if field.get_val(x+2, y+2) == op { pr3 -= 1; } else { pr2 -= 1; } }
						if field.get_val(x, y+1) == op { if field.get_val(x, y+2) == op { pr3 -= 1; } else { pr2 -= 1; } }
						if x>=1 && field.get_val(x-1, y+1) == op { if x>=2 && field.get_val(x-2, y+2) == op { pr3 -= 1; } else { pr2 -= 1; } }
						//free squares
						if y>=1
						{
							if x>=1 && field.get_val(x-1, y-1) == 0 { heur -= value; }
							if field.get_val(x, y-1) == 0 { heur -= value; }
							if x<field.get_w()-1 && field.get_val(x+1, y-1) == 0 { heur -= value; }
						}
						if x>=1 && field.get_val(x-1, y) == 0 { heur -= value; }
						if x<field.get_w()-1 && field.get_val(x+1, y) == 0 { heur -= value; }
						if y<field.get_h()-1
						{
							if x>=1 && field.get_val(x-1, y+1) == 0 { heur -= value; }
							if x<field.get_w()-1 && field.get_val(x+1, y+1) == 0 { heur -= value; }
						}
					}
				}
			}
			heur += (2*pr2 + 5*pr3) as f64; //add up to final score (3 rows count far more than 2 rows)
			return heur;
		}
	}
	
	fn minimax(field:&mut Field, p:i32, deep:u32) -> f64
	{
		let op = if p == 1 {2} else {1};
		if deep > DEEPNESS { return PlayerMinimax::heur(field, if deep%2 == 0 {op} else {p}, deep); } //leaf node -> return evaluated heuristic
		let state = field.get_state(); //return early on game end
		if state == -1 { return 0.0; }
		else if state == p { return if deep%2 == 0 {-10002.0 + deep as f64} else {10002.0 - deep as f64}; }
		else if state == op { return if deep%2 == 0 {10002.0 - deep as f64} else {-10002.0 + deep as f64}; }
		
		//else: game running -> go deeper
		let mut heur = if deep%2 == 0 { f64::INFINITY } else { f64::NEG_INFINITY };
		for i in 0..field.get_w()
		{
			if field.is_valid_play(i)
			{
				field.play(p, i);
				let val = PlayerMinimax::minimax(field, op, deep+1);
				field.undo();
				if (deep%2 == 0 && val < heur) || (deep%2 == 1 && val > heur)
				{
					heur = val;
				}
			}
		}
		heur
	}
}

impl Player for PlayerMinimax
{
	#[allow(unused_variables)]
	fn init(&mut self, field:&Field, p:i32) -> bool
	{
		if DEEPNESS < 1 { return false; }
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
		
		let p = self.pid;
		let op = if p == 1 {2} else {1};
		let mut handles = Vec::new();
		//spawn threads (one for each choice)
		for i in 0..field.get_w()
		{
			let mut pfield = field.clone();
			handles.push(thread::spawn(move ||
				{
					if pfield.play(p, i) { PlayerMinimax::minimax(&mut pfield, op, 2) }
					else { f64::NEG_INFINITY }
					//undo not needed, because it was cloned and will be dropped
				}));
		}
		
		//decide which action to take
		let mut x:u32 = 0;
		let mut max = f64::NEG_INFINITY;
		for i in 0..field.get_w()
		{
			let res = handles.pop().unwrap().join();
			if res.is_err() { return false; }
			let res = res.unwrap();
			if max < res || !field.is_valid_play(x)
			{
				max = res;
				x = field.get_w()-i-1;
			}
		}
		
		//debug
		//println!("Heur: {}", max);
		
		//play (actually should always be true, unless game was finished before)
		field.play(self.pid, x)
	}
	
	#[allow(unused_variables)]
	fn outcome(&mut self, field:&mut Field, state:i32)
	{
		//nothing
	}
}

impl Drop for PlayerMinimax
{
	fn drop(&mut self)
	{
		//nothing to do
	}
}
