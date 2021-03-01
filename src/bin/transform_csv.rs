use portals::joined_by::*;

fn main() {
	let file_input = std::fs::read_to_string("points30_1000.csv").unwrap();
	use std::{fs::File, io::prelude::*};
	let mut file = File::create("points30_1000_sincos_poly.csv").unwrap();
	for line in file_input.split('\n').filter(|x| !x.trim().is_empty()) {
		let line = line.split(',').map(|x| x.parse::<f64>().unwrap()).collect::<Vec<_>>();
		let a = line[0];
		let b = line[1];
		let c = line[2];
		let d = line[3];

		let result = line[4];

		let mut array = Vec::new();

		array.push(a);
		array.push(b);
		array.push(c);
		array.push(d);

		array.push(a.sin());
		array.push(b.sin());
		array.push(c.sin());
		array.push(d.sin());

		array.push(a.sin() * a.sin());
		array.push(b.sin() * a.sin());
		array.push(c.sin() * a.sin());
		array.push(d.sin() * a.sin());

		array.push(a.sin() * b.sin());
		array.push(b.sin() * b.sin());
		array.push(c.sin() * b.sin());
		array.push(d.sin() * b.sin());

		array.push(a.sin() * c.sin());
		array.push(b.sin() * c.sin());
		array.push(c.sin() * c.sin());
		array.push(d.sin() * c.sin());

		array.push(a.sin() * d.sin());
		array.push(b.sin() * d.sin());
		array.push(c.sin() * d.sin());
		array.push(d.sin() * d.sin());

		array.push(a.cos());
		array.push(b.cos());
		array.push(c.cos());
		array.push(d.cos());

		array.push(a.cos() * a.cos());
		array.push(b.cos() * a.cos());
		array.push(c.cos() * a.cos());
		array.push(d.cos() * a.cos());

		array.push(a.cos() * b.cos());
		array.push(b.cos() * b.cos());
		array.push(c.cos() * b.cos());
		array.push(d.cos() * b.cos());

		array.push(a.cos() * c.cos());
		array.push(b.cos() * c.cos());
		array.push(c.cos() * c.cos());
		array.push(d.cos() * c.cos());

		array.push(a.cos() * d.cos());
		array.push(b.cos() * d.cos());
		array.push(c.cos() * d.cos());
		array.push(d.cos() * d.cos());

		array.push(result);

		writeln!(file, "{}", array.iter().joined_by(",")).unwrap();
	}
}
