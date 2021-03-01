use core::f64::consts::PI;

use indicatif::{ProgressBar, ProgressStyle};
use portals::{
	angle::Radians,
	assert_between,
	image::Image,
	joined_by::*,
	mobius::MobiusPoints,
	sphere::{Angles3, SphereLine3},
};
use rand::prelude::*;

fn make_data_to_improved_predictions(points: &MobiusPoints) {
	let mut rng = rand::thread_rng();

	use std::{fs::File, io::prelude::*};
	let mut file = File::create("points.csv").unwrap();

	let sphere = MobiusPoints::sphere();

	let size = 300_000;

	let bar = ProgressBar::new(size as u64).with_style(
		ProgressStyle::default_bar()
			.template("[elapsed: {elapsed_precise:>8} | remaining: {eta_precise:>8} | {percent:>3}%] {wide_bar}"),
	);

	for _ in 0..size {
		bar.inc(1);
		let alpha1 = Radians(rng.gen_range((0.)..2. * PI));
		let beta1 = Radians(rng.gen_range((0.)..PI));
		let alpha2 = Radians(rng.gen_range((0.)..2. * PI));
		let beta2 = Radians(rng.gen_range((0.)..PI));

		let o = Angles3::new(alpha1, beta1);
		let od = Angles3::new(alpha2, beta2);
		let sphere_line = SphereLine3 { o, od };

		if let Some(should_be) = MobiusPoints::f(&sphere, &sphere_line, 100) {
			let anglesi = points.solved_points.get_integer_angles(&sphere_line);
			let current = points.solved_points[anglesi].map(|x| x.0).unwrap_or(-1.);

			writeln!(
				file,
				"{}",
				[alpha1.0, beta1.0, alpha2.0, beta2.0, current, should_be.best_u.0]
					.iter()
					.joined_by(",")
			)
			.unwrap();
		}
	}
}

fn main() {
	let count = 20;
	// make_data_to_improved_predictions(&MobiusPoints::load());
	let points = MobiusPoints::calc(count);
	// let points = MobiusPoints::load("points30_1000.bin");
	// points.save_csv("points30_1000.csv");
	// points.save("points30_1000.bin");
	points.save_to_texture("data/texture10_30.png");
	// let points = MobiusPoints::load();
	// points.save_csv();
}
