#![allow(dead_code)]

#[derive(Clone)]
pub struct Field
{
	field: Vec<i32>,
	w: u32,
	h: u32,
	turns: Vec<(u32, u32)>,
}

impl Field
{
	pub fn new(width:u32, height:u32) -> Field
	{
		let size:usize = (width*height) as usize;
		Field { field:vec![0i32; size], w:width, h:height, turns: Vec::new() }
	}
	
	pub fn get_w(&self) -> u32
	{
		self.w
	}
	
	pub fn get_h(&self) -> u32
	{
		self.h
	}
	
	pub fn get_size(&self) -> u32
	{
		self.w*self.h //self.field.len()
	}
	
	pub fn get_field(&self) -> &Vec<i32>
	{
		&self.field
	}
	
	pub fn get_turns(&self) -> &Vec<(u32, u32)>
	{
		&self.turns
	}
	
	pub fn get_val(&self, x:u32, y:u32) -> i32
	{
		if x<self.w && y<self.h
		{
			self.field[(y*self.w+x) as usize]
		}
		else
		{
			0
		}
	}
	
	fn set_val(&mut self, x:u32, y:u32, val:i32)
	{
		if x<self.w && y<self.h
		{
			self.field[(y*self.w+x) as usize] = val;
		}
	}
	
	pub fn reset(&mut self)
	{
		for i in 0..self.field.len()
		{
			self.field[i] = 0;
		}
		self.turns.clear();
	}
	
	pub fn play(&mut self, player:i32, x:u32) -> bool
	{
		if x >= self.w || player < 1 || player > 2 || !self.is_valid_play(x)
		{
			return false;
		}
		
		for y in (0..self.h).rev()
		{
			if self.get_val(x, y) == 0
			{
				self.set_val(x, y, player);
				self.turns.push((x, y));
				return true;
			}
		}
		
		false
	}
	
	pub fn undo(&mut self) -> bool
	{
		match self.turns.pop()
		{
			Some((x, y)) => { self.set_val(x, y, 0); true },
			_ => false
		}
	}
	
	pub fn print(&self)
	{
		for y in 0..self.h
		{
			let mut str = " ".to_string();
			let mut strline = "".to_string();
			for x in 0..self.w
			{
				//str += &(self.get_val(x, y).to_string()); //as number
				str += match self.get_val(x, y)
					{
						1 => "X",
						2 => "O",
						_ => " ",
					}; // as XO string
				str += " | ";
				strline += "----";
			}
			println!("{}", &str[..(str.len()-2)]);
			if y < self.h-1
			{
				println!("{}", &strline[..(strline.len()-1)]);
				}
		}
	}
	
	pub fn is_valid_play(&self, x:u32) -> bool
	{
		x < self.w && self.get_val(x, 0) == 0
	}
	
	pub fn is_full(&self) -> bool
	{
		self.turns.len() == self.field.len()
	}
	
	/// return: 0=running, -1=draw, 1=player1 win, 2=player2 win
	pub fn get_state(&self) -> i32
	{
		for y in 0..self.h
		{
			for x in 0..self.w
			{
				if self.get_val(x, y) == 0 { continue; }
				
				if (self.get_val(x, y) == self.get_val(x+1, y) && self.get_val(x+1, y) == self.get_val(x+2, y) && self.get_val(x+2, y) == self.get_val(x+3, y)) ||
					(self.get_val(x, y) == self.get_val(x+1, y+1) && self.get_val(x+1, y+1) == self.get_val(x+2, y+2) && self.get_val(x+2, y+2) == self.get_val(x+3, y+3)) ||
					(self.get_val(x, y) == self.get_val(x, y+1) && self.get_val(x, y+1) == self.get_val(x, y+2) && self.get_val(x, y+2) == self.get_val(x, y+3)) ||
					(x >= 3 && self.get_val(x, y) == self.get_val(x-1, y+1) && self.get_val(x-1, y+1) == self.get_val(x-2, y+2) && self.get_val(x-2, y+2) == self.get_val(x-3, y+3))
				{
					match self.get_val(x, y)
					{
						1 => { return 1; },
						2 => { return 2; },
						_ => { continue; },
					}
				}
			}
		}
		
		if self.is_full() { return -1; }
		0
	}
}
