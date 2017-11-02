extern crate ernn;

use self::ernn::*;
use game::*;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::io::prelude::*;
use std::time::Instant;


#[allow(dead_code, unused_variables)]
pub fn main()
{
	let filename1 = "AIValue-7x6.NN";
	let filename2 = "AIValue-7x6-test.NN";
	train(filename1, 5, 2);
	play(filename1);
	//battle(filename1, filename2);
} //TODO: try optimizer parameters, save evaluator

#[allow(dead_code)]
pub fn play(filename:&str)
{
	let (_, nn) = load_nn(filename);
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(PlayerType::IO);
	game.set_player2_nn(PlayerType::AIValue, nn);
	println!("Player X: IO");
	println!("Player O: AIValue");
	println!("");
	let (w, d, l) = game.play_many(2, 1);
	println!("Results:");
	println!("Player X wins: {:>6.2}%", w);
	println!("Draws:         {:>6.2}%", d);
	println!("Player O wins: {:>6.2}%", l);
}

#[allow(dead_code)]
pub fn battle(filename1:&str, filename2:&str)
{
	let (_, nn1) = load_nn(filename1);
	let (_, nn2) = load_nn(filename2);
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1_nn(PlayerType::AIValue, nn1);
	game.set_player2_nn(PlayerType::AIValue, nn2);
	println!("Player 1: {}", filename1);
	println!("Player 2: {}", filename2);
	println!("");
	let (w, d, l) = game.play_many(2, 1);
	println!("Results:");
	println!("Player 1 wins: {:>6.2}%", w);
	println!("Draws:         {:>6.2}%", d);
	println!("Player 2 wins: {:>6.2}%", l);
}

#[allow(dead_code)]
pub fn train(filename:&str, rounds:u32, gens:u32)
{
	//parameters for optimizer
	let population = 40;
	let survival = 10;
	let badsurv = 5;
	let prob_avg = 0.1;
	let prob_mut = 0.95;
	let prob_op = 0.1;
	let op_range = 0.2;
	let prob_block = 0.2;
	let prob_new = 0.05;
	
	//init NN and optimizer
	let (mut num_gens, mut nn) = load_nn(filename);
	let mut eval = AIValueEval::new(nn.clone());
	let mut opt = Optimizer::new(eval.clone(), nn);
	let mut score = opt.optimize(1, survival+badsurv, survival, badsurv, 0.0, 1.0, prob_op, op_range, prob_block, 0.5); //initial population
	println!("Starting score: {}", score);
	
	//optimize
	let now = Instant::now();
	for _ in 0..rounds
	{
		score = opt.optimize(gens, population, survival, badsurv, prob_avg, prob_mut, prob_op, op_range, prob_block, prob_new);
		println!("Generation score: {}", score);
		nn = opt.get_nn();
		eval.add_cmp(nn); //moved
		opt.set_eval(eval.clone());
		score = opt.reevaluate();
		//save NN
		num_gens += gens;
		nn = opt.get_nn();
		save_nn(filename, num_gens, &nn);
	}
	println!("End score: {}", score);
	let elapsed = now.elapsed();
	let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	
	//print information
	nn = opt.get_nn();
	println!("NN blocks: {}", nn.get_blocks());
	println!("NN Gen/Opt Gen: {}/{}", nn.get_gen(), num_gens);
}

#[allow(dead_code)]
fn load_nn(filename:&str) -> (u32, NN)
{
	let nn;
	let num_gens;
	let file = File::open(filename);
	if file.is_err()
	{
		//create new neural net, as it could not be loaded
		let n = 7*6;
		nn = NN::new(n, 2*n, 1, Activation::PELU, Activation::Tanh); //set NN arch here
		num_gens = 0;
	}
	else
	{
		//load neural net from file (and number of generations)
		let mut reader = BufReader::new(file.unwrap());
		let mut datas = String::new();
		let mut nns = String::new();
		
		let res1 = reader.read_line(&mut datas);
		let res2 = reader.read_line(&mut nns);
		if res1.is_err() || res2.is_err() { panic!("Error reading AIValue NN file!"); }
		
		let res = datas.trim().parse::<u32>();
		if res.is_err() { panic!("Error parsing AIValue NN file!"); }
		num_gens = res.unwrap();
		nn = NN::from_json(&nns);
	}
	//return
	(num_gens, nn)
}

#[allow(dead_code)]
fn save_nn(filename:&str, num_gens:u32, nn:&NN)
{
	//write neural net to file
	let file = File::create(filename);
	if file.is_err() { eprintln!("Warning: Could not write AIValue NN file!"); return; }
	let mut writer = BufWriter::new(file.unwrap());
	
	let res1 = writeln!(&mut writer, "{}", num_gens);
	let res2 = writeln!(&mut writer, "{}", nn.to_json());
	if res1.is_err() || res2.is_err() { eprintln!("Warning: There was an error while writing AIValue NN file!"); return; }
}



const NUM_CMP:usize = 10;
#[derive(Clone)]
struct AIValueEval
{
	curr_cmp: Vec<NN>,
}

impl AIValueEval
{
	pub fn new(nn:NN) -> AIValueEval
	{
		let mut nnvec = Vec::with_capacity(NUM_CMP);
		nnvec.push(nn);
		AIValueEval { curr_cmp: nnvec }
	}
	
	pub fn add_cmp(&mut self, nn:NN)
	{
		if self.curr_cmp.len() >= NUM_CMP
		{
			self.curr_cmp.remove(1);
		}
		self.curr_cmp.push(nn);
	}
}

impl Evaluator for AIValueEval
{
	fn evaluate(&self, nn:&NN) -> f64
	{
		let mut g = Game::new();
		g.set_start_player(1);
		g.set_player2_nn(PlayerType::AIValue, nn.clone());
		//play against random
		g.set_player1(PlayerType::Random);
		let (_, _, r) = g.play_many(250, 1); //1000?
		//play against minimax
		g.set_player1(PlayerType::Minimax);
		let (_, _, m) = g.play_many(2, 1);
		//play against cmp net
		let mut c = 0.0;
		for nn in &self.curr_cmp
		{
			g.set_player1_nn(PlayerType::AIValue, nn.clone());
			let (_, _, cl) = g.play_many(2, 1);
			c += cl;
		}
		c /= self.curr_cmp.len() as f64;
		//score
		let mut score = (r - 50.0) * 10.0; //better than random, but higher weight
		score += m; //win against minimax
		score += c / 50.0; //better than previous self versions, but not too strong weight
		//return
		score
	}
}
