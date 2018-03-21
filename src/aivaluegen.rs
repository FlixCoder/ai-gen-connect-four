extern crate ernn;

use self::ernn::*;
use game::*;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::io::prelude::*;
use std::time::Instant;
use super::serde_json;


#[allow(dead_code, unused_variables)]
pub fn main()
{
	let filename1 = "AIValue-7x6.NN";
	let filename2 = "AIValue-7x6-bak.NN";
	
	train(filename1, 25, 2, true);
	play(filename1);
	battle(filename2, filename1);
	test_minimax(filename1);
	test_random(filename1);
	print_info(filename1);
}

#[allow(dead_code)]
pub fn play(filename:&str)
{
	let (_, nn, _, _) = load_nn(filename);
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(PlayerType::IO);
	game.set_player2_nn(PlayerType::AIValue, nn);
	println!("Player X: IO");
	println!("Player O: AIValue");
	println!("");
	let (w, d, l) = game.play_many(2, 1);
	println!("");
	println!("Results:");
	println!("Player X wins: {:>6.2}%", w);
	println!("Draws:         {:>6.2}%", d);
	println!("Player O wins: {:>6.2}%", l);
	println!("");
}

#[allow(dead_code)]
pub fn battle(filename1:&str, filename2:&str)
{
	let (_, nn1, _, _) = load_nn(filename1);
	let (_, nn2, _, _) = load_nn(filename2);
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1_nn(PlayerType::AIValue, nn1);
	game.set_player2_nn(PlayerType::AIValue, nn2);
	println!("Player 1: AIValue {}", filename1);
	println!("Player 2: AIValue {}", filename2);
	let (w, d, l) = game.play_many(2, 1);
	println!("Results:");
	println!("Player 1 wins: {:>6.2}%", w);
	println!("Draws:         {:>6.2}%", d);
	println!("Player 2 wins: {:>6.2}%", l);
	println!("");
}

#[allow(dead_code)]
pub fn test_minimax(filename:&str)
{
	let (_, nn, _, _) = load_nn(filename);
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(PlayerType::Minimax);
	game.set_player2_nn(PlayerType::AIValue, nn);
	println!("Player 1: Minimax");
	println!("Player 2: AIValue {}", filename);
	let (w, d, l) = game.play_many(2, 1);
	println!("Results:");
	println!("Player 1 wins: {:>6.2}%", w);
	println!("Draws:         {:>6.2}%", d);
	println!("Player 2 wins: {:>6.2}%", l);
	println!("");
}

#[allow(dead_code)]
pub fn test_random(filename:&str)
{
	let (_, nn, _, _) = load_nn(filename);
	let mut game = Game::new();
	game.set_start_player(1);
	game.set_player1(PlayerType::Random);
	game.set_player2_nn(PlayerType::AIValue, nn);
	println!("Player 1: Random");
	println!("Player 2: AIValue {}", filename);
	let (w, d, l) = game.play_many(1000, 1);
	println!("Results:");
	println!("Player 1 wins: {:>6.2}%", w);
	println!("Draws:         {:>6.2}%", d);
	println!("Player 2 wins: {:>6.2}%", l);
	println!("");
}

#[allow(dead_code)]
pub fn print_info(filename:&str)
{
	let (num_gens, nn, _, opt) = load_nn(filename);
	println!("NN score: {}", opt.get_population()[0].1);
	println!("NN blocks: {}", nn.get_blocks());
	println!("NN Gen/Opt Gen: {}/{}", nn.get_gen(), num_gens);
	println!("");
}


const HIDDEN:u32 = 10; //hidden layers' size
const NUM_CMP:usize = 100; //number of NNs to keep for comparison in the evaluator to evaluate new NNs
#[allow(dead_code)]
pub fn train(filename:&str, rounds:u32, gens:u32, par:bool)
{
	//parameters for optimizer
	let population = 200;
	let survival = 4;
	let badsurv = 1;
	let prob_avg = 0.1; //keep
	let prob_mut = 0.95; //keep
	let prob_new = 0.1; //keep
	let prob_block = 0.02;
	let prob_op = 0.2;
	let op_range = 0.1;
	
	//init NN and optimizer
	let (mut num_gens, _, mut eval, mut opt) = load_nn(filename);
	let mut score;
	if num_gens == 0
	{
		score = opt.gen_population(survival+badsurv); //initial population
	}
	else
	{
		score = opt.reevaluate(); //_par?
	}
	println!("Starting score: {}", score);
	
	//optimize
	let now = Instant::now();
	for _ in 0..rounds
	{
		if par { score = opt.optimize_par(gens, population, survival, badsurv, prob_avg, prob_mut, prob_new, prob_block, prob_op, op_range); }
		else { score = opt.optimize(gens, population, survival, badsurv, prob_avg, prob_mut, prob_new, prob_block, prob_op, op_range); }
		println!("Generation score: {}", score);
		let nn = opt.get_nn();
		eval.add_cmp(nn); //nn moved into func
		opt.set_eval(eval.clone());
		if par { score = opt.reevaluate_par(); }
		else { score = opt.reevaluate(); }
		//save NN
		num_gens += gens;
		save_nn(filename, num_gens, &eval, &opt);
	}
	println!("End score: {}", score);
	let elapsed = now.elapsed();
	let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	
	//print information
	let nn = opt.get_nn();
	println!("NN blocks: {}", nn.get_blocks());
	println!("NN Gen/Opt Gen: {}/{}", nn.get_gen(), num_gens);
	println!("");
}

#[allow(dead_code)]
fn load_nn(filename:&str) -> (u32, NN, AIValueEval, Optimizer<AIValueEval>)
{
	let num_gens;
	let nn;
	let eval;
	let mut opt;
	
	let n = 7*6;
	let file = File::open(filename);
	if file.is_err()
	{
		//create new neural net, as it could not be loaded
		num_gens = 0;
		let nn = NN::new(n, HIDDEN, 1, Activation::RELU, Activation::Tanh); //set NN arch here
		eval = AIValueEval::new(nn.clone());
		opt = Optimizer::new(eval.clone(), nn);
	}
	else
	{
		//load neural net from file (and number of generations)
		let mut reader = BufReader::new(file.unwrap());
		let mut datas = String::new();
		let mut evals = String::new();
		let mut opts = String::new();
		
		let res1 = reader.read_line(&mut datas);
		let res2 = reader.read_line(&mut evals);
		let res3 = reader.read_line(&mut opts);
		if res1.is_err() || res2.is_err() || res3.is_err() { panic!("Error reading AIValue NN file!"); }
		
		let res = datas.trim().parse::<u32>();
		if res.is_err() { panic!("Error parsing AIValue NN file!"); }
		num_gens = res.unwrap();
		eval = AIValueEval::from_json(&evals);
		opt = Optimizer::new(eval.clone(), NN::new(n, 1, 1, Activation::Linear, Activation::Tanh)); //NN will get dropped in load_population (but evaluated once)
		opt.load_population(&opts);
	}
	//return
	nn = opt.get_nn();
	(num_gens, nn, eval, opt)
}

#[allow(dead_code)]
fn save_nn(filename:&str, num_gens:u32, eval:&AIValueEval, opt:&Optimizer<AIValueEval>)
{
	//write neural net to file
	let file = File::create(filename);
	if file.is_err() { eprintln!("Warning: Could not write AIValue NN file!"); return; }
	let mut writer = BufWriter::new(file.unwrap());
	
	let res1 = writeln!(&mut writer, "{}", num_gens);
	let res2 = writeln!(&mut writer, "{}", eval.to_json());
	let res3 = writeln!(&mut writer, "{}", opt.save_population());
	if res1.is_err() || res2.is_err() || res3.is_err() { eprintln!("Warning: There was an error while writing AIValue NN file!"); return; }
}



#[derive(Clone, Deserialize, Serialize)]
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
		let len = self.curr_cmp.len();
		if self.curr_cmp[len - 1] != nn
		{ //add only if the same net is not already in there (just check last added net)
			if len >= NUM_CMP
			{
				self.curr_cmp.remove(0);
			}
			self.curr_cmp.push(nn);
		}
	}
	
	/// Encodes the evaluator as a JSON string.
    pub fn to_json(&self) -> String
	{
        serde_json::to_string(self).ok().expect("Encoding JSON failed!")
    }

	/// Builds a new evaluator from a JSON string.
    pub fn from_json(encoded: &str) -> AIValueEval
	{
        let network:AIValueEval = serde_json::from_str(encoded).ok().expect("Decoding JSON failed!");
        network
    }
}

impl Evaluator for AIValueEval
{
	/// Evaluates the neural net as how strong it is against a random, minimax or previous self player it is
	/// Prefers random and minimax value to stabilize training and avoid overestimation in self play
	/// Gains value from wins against previous self version to implement constant improvement
	fn evaluate(&self, nn:&NN) -> f64
	{
		let mut g = Game::new();
		g.set_start_player(1);
		g.set_player2_nn(PlayerType::AIValue, nn.clone());
		//play against minimax
		g.set_player1(PlayerType::Minimax);
		let (_, d, mut m) = g.play_many(2, 1);
		m += d / 2.0; //add draws as half
		//play against random
		g.set_player1(PlayerType::Random);
		let (_, d, mut r) = g.play_many(1000, 1);
		r += d / 2.0; //add draws as half
		//play against cmp net
		let mut c = 0.0;
		for nn in &self.curr_cmp
		{
			g.set_player1_nn(PlayerType::AIValue, nn.clone());
			let (_, d, cl) = g.play_many(2, 1);
			c += cl + d / 2.0; //add draws as half
		}
		c /= self.curr_cmp.len() as f64;
		//score
		let mut score = m * 10.0; //betterness against minimax, adjusted weight
		score += r * 10.0; //betternes against random, adjusted weight
		score += c / 10.0; //betterness against previous self versions, adjusted weight, so weight matches random deviation //bak: 10.0
		//return
		score
	}
}
