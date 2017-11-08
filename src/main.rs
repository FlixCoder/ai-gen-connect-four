#[macro_use]
extern crate serde_derive;

extern crate serde;
extern crate serde_json;

mod game;
mod aivaluegen;


fn main()
{
	aivaluegen::main();
}


