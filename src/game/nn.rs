//! Modified NN, originally from: https://github.com/jackm321/RustNN

//to root:
//#[macro_use]
//extern crate serde_derive;

extern crate serde;
extern crate serde_json;
extern crate rand;

use std::iter::{Zip, Enumerate};
use std::slice;
use self::rand::Rng;
use self::rand::distributions::{Normal, IndependentSample};

const DEFAULT_LEARNING_RATE:f64 = 0.3;
const DEFAULT_LAMBDA:f64 = 0.0;
const DEFAULT_MOMENTUM:f64 = 0.0;
const DEFAULT_EPOCHS:u32 = 1000;

//values for a (0,1) distribution (so (-1, 1) interval in standard deviation)
//const SELU_FACTOR_A:f64 = 1.0507; //greater than 1, lambda in https://arxiv.org/pdf/1706.02515.pdf
//const SELU_FACTOR_B:f64 = 1.6733; //alpha in https://arxiv.org/pdf/1706.02515.pdf
//values for a (0,2) distribution (so (-2, 2) interval in standard deviation)
const SELU_FACTOR_A:f64 = 1.06071; //greater than 1, lambda in https://arxiv.org/pdf/1706.02515.pdf
const SELU_FACTOR_B:f64 = 1.97126; //alpha in https://arxiv.org/pdf/1706.02515.pdf

const PELU_FACTOR_A:f64 = 1.5;
const PELU_FACTOR_B:f64 = 2.0;

const LRELU_FACTOR:f64 = 0.33;


/// Specifies the activation function
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Activation
{
	/// Sigmoid activation
	Sigmoid,
	/// SELU activation
	SELU,
	/// PELU activation
	PELU,
	/// Leaky ReLU activation
	LRELU,
	/// Linear activation
	Linear,
	/// Tanh activation
	Tanh,
}

/// Neural network
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NN
{
    layers: Vec<Vec<Vec<f64>>>, //weights
    num_inputs: u32, //number of inputs to NN
	hidden_size: u32, //size of every hidden layer. always 2 layers in a block. at least 1 hidden layer
	hid_act: u32, //hidden layer activation
	out_act: u32, //output layer activation
	blocks: u32, //number of residual layer blocks
	generation: u32, //generation of current network
}

impl NN
{
    pub fn new(inputs:u32, hidden_size:u32, outputs:u32, hidden_activation:Activation, output_activation:Activation) -> NN
	{
        let mut rng = rand::thread_rng();

        if inputs < 1 || hidden_size < 1 || outputs < 1
		{
            panic!("inappropriate parameter bounds");
        }

		// setup the layers
        let mut layers = Vec::new();
        let mut prev_layer_size = inputs;
        for i in 0..2
		{ //one hidden layer and one output layer
            let mut layer: Vec<Vec<f64>> = Vec::new();
			let layer_size = if i == 1 { outputs } else { hidden_size };
			let mut init_std_scale = 2.0; //He init
			if hidden_activation == Activation::SELU { init_std_scale = 1.0; } //MSRA / Xavier init
			let normal = Normal::new(0.0, (init_std_scale / prev_layer_size as f64).sqrt());
            for _ in 0..layer_size
			{
                let mut node: Vec<f64> = Vec::with_capacity(1 + prev_layer_size as usize);
                for i in 0..prev_layer_size+1
				{
					if i == 0 //threshold aka bias
					{
						node.push(0.0);
					}
					else
					{
						let random_weight: f64 = normal.ind_sample(&mut rng);
						node.push(random_weight);
					}
                }
                layer.push(node)
            }
            layer.shrink_to_fit();
            layers.push(layer);
            prev_layer_size = layer_size;
        }
        layers.shrink_to_fit();
		
		//set activation functions
		let hid_act = match hidden_activation {
			Activation::Sigmoid => 0,
			Activation::SELU => 1,
			Activation::PELU => 2,
			Activation::LRELU => 3,
			Activation::Linear => 4,
			Activation::Tanh => 5,
		};
		let out_act = match output_activation {
			Activation::Sigmoid => 0,
			Activation::SELU => 1,
			Activation::PELU => 2,
			Activation::LRELU => 3,
			Activation::Linear => 4,
			Activation::Tanh => 5,
		};
        NN { layers: layers, num_inputs: inputs, hidden_size: hidden_size, hid_act: hid_act, out_act: out_act, blocks: 0, generation: 0 }
    }
	
    pub fn run(&self, inputs: &[f64]) -> Vec<f64>
	{
        if inputs.len() as u32 != self.num_inputs
		{
            panic!("input has a different length than the network's input layer");
        }
        self.do_run(inputs).pop().unwrap()
    }
	
    /// Encodes the network as a JSON string.
    pub fn to_json(&self) -> String
	{
        serde_json::to_string(self).ok().expect("encoding JSON failed!")
    }

    /// Builds a new network from a JSON string.
    pub fn from_json(encoded: &str) -> NN
	{
        let network: NN = serde_json::from_str(encoded).ok().expect("Decoding JSON failed!");
        network
    }
	
    fn do_run(&self, inputs: &[f64]) -> Vec<Vec<f64>>
	{
        let mut results = Vec::new();
        results.push(inputs.to_vec());
		let num_layers = self.layers.len();
        for (layer_index, layer) in self.layers.iter().enumerate()
		{
            let mut layer_results = Vec::new();
            for (i, node) in layer.iter().enumerate()
			{
				let mut sum = modified_dotprod(&node, &results[layer_index]); //sum of forward pass to this node
				//residual network shortcut
				if layer_index >= 1 && layer_index < num_layers - 1 && layer_index % 2 == 0
				{
					sum += results[layer_index - 1][i];
				}
				//standard forward pass activation
				layer_results.push( if layer_index == self.layers.len()-1 //output layer
					{
						match self.out_act {
							0 => sigmoid(sum), //sigmoid
							1 => selu(sum), //selu
							2 => pelu(sum), //pelu
							3 => lrelu(sum), //lrelu
							4 => linear(sum), //linear
							_ => tanh(sum), //tanh
						}
					}
					else
					{
						match self.hid_act {
							0 => sigmoid(sum), //sigmoid
							1 => selu(sum), //selu
							2 => pelu(sum), //pelu
							3 => lrelu(sum), //lrelu
							4 => linear(sum), //linear
							_ => tanh(sum), //tanh
						}
					} );
            }
            results.push(layer_results);
        }
        results
    }
	
    //generate mutated version of current network
	//ideas to add: change activation function or at least activation function parameters,
	//			still use backprop for something to speed up calculation
	//			sometimes return a completely fresh initialized network?
	//params: (all probabilities in [0,1])
	//prob_op:f64 - probability to apply an addition/substraction to a node
	//op_range:f64 - maximum positive or negative adjustment of a weight
	//prob_block:f64 - probability to add another residual block (2 layers) somewhere in the network, initially identity, random prob_op afterwards
    pub fn generate_mutation(&self, prob_op:f64, op_range:f64, prob_block:f64)
	{
		let mut rng = rand::thread_rng();
		let mut newnn = self.clone();
		newnn.increment_generation();
		//random residual block addition
		if rng.gen::<f64>() < prob_block
		{
			newnn.mutate_block();
		}
		//random addition / substraction op
		if prob_op != 0.0 && op_range != 0.0
		{
			newnn.mutate_op(prob_op, op_range);
		}
    }
	
	fn increment_generation(&mut self)
	{
		self.generation += 1;
	}
	
	fn mutate_block(&mut self)
	{
		let mut rng = rand::thread_rng();
		let mut place:usize = 1; //index of layer where to put the new block
		if self.blocks != 0
		{
			place += 2 * (rng.gen::<usize>() % (1 + self.blocks as usize));
		}
		
		//insert block
		let mut layer1:Vec<Vec<f64>> = Vec::with_capacity(self.hidden_size as usize);
		let mut layer2:Vec<Vec<f64>> = Vec::with_capacity(self.hidden_size as usize);
		for _ in 0..self.hidden_size
		{
			layer1.push(vec![0.0; 1 + self.hidden_size as usize]);
			layer2.push(vec![0.0; 1 + self.hidden_size as usize]);
		}
		self.layers.insert(place, layer2);
		self.layers.insert(place, layer1);
		
		self.blocks += 1;
	}
	
	fn mutate_op(&mut self, prob_op:f64, op_range:f64)
	{
		let mut rng = rand::thread_rng();
		for layer_index in 0..self.layers.len()
		{
            let mut layer = &mut self.layers[layer_index];
            for node_index in 0..layer.len()
			{
                let mut node = &mut layer[node_index];
                for weight_index in 0..node.len()
				{
                    let mut delta = 0.0;
					if rng.gen::<f64>() < prob_op
					{
						delta = op_range * (2.0 * rng.gen::<f64>() - 1.0);
					}
                    node[weight_index] += delta;
                }
            }
        }
	}
}
	
fn sigmoid(y: f64) -> f64
{
    1f64 / (1f64 + (-y).exp())
}

fn selu(y: f64) -> f64
{ //SELU activation
	SELU_FACTOR_A * if y < 0.0
	{
		SELU_FACTOR_B * y.exp() - SELU_FACTOR_B
	}
	else
	{
		y
	}
}

fn pelu(y: f64) -> f64
{ //PELU activation
	if y < 0.0
	{
		PELU_FACTOR_A * (y / PELU_FACTOR_B).exp() - PELU_FACTOR_A
	}
	else
	{
		(PELU_FACTOR_A / PELU_FACTOR_B) * y
	}
}

fn lrelu(y: f64) -> f64
{ //LRELU activation
	if y < 0.0
	{
		LRELU_FACTOR * y
	}
	else
	{
		y
	}
}

fn linear(y: f64) -> f64
{ //linear activation
	y
}

fn tanh(y: f64) -> f64
{ //tanh activation
	y.tanh()
}


fn modified_dotprod(node: &Vec<f64>, values: &Vec<f64>) -> f64
{
    let mut it = node.iter();
    let mut total = *it.next().unwrap(); // start with the threshold weight
    for (weight, value) in it.zip(values.iter())
	{
        total += weight * value;
    }
    total
}

// takes two arrays and enumerates the iterator produced by zipping each of
// their iterators together
fn iter_zip_enum<'s, 't, S: 's, T: 't>(s: &'s [S], t: &'t [T]) ->
    Enumerate<Zip<slice::Iter<'s, S>, slice::Iter<'t, T>>>
{
    s.iter().zip(t.iter()).enumerate()
}
