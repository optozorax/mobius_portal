use core::f64::consts::PI;
use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3 as Vec3;
use portals::{
	angle::Degrees,
	assert_between,
	cam::Cam,
	image::Image,
	mobius::{calc_mobius_everywhere, MobiusPoints},
	quadratic_equation::Roots,
	sphere::Sph3,
};

fn main() {
	// let mut image = Image::new(300, 300);
	// let cam = Cam::look_at_from(
	// 	Vec3::new(0., 0., 0.),
	// 	Sph3::new(Degrees(20.).into(), Degrees(1.).into(), 2.5).into(),
	// 	Degrees(80.).into(),
	// );
	// let sphere = MobiusPoints::sphere();

	// for (x, y, pos) in image.iter() {
	// 	let ray = cam.get_ray(pos);
	// 	let color = sphere
	// 		.to_line(&ray)
	// 		.map(|(l, _)| {
	// 			// let ray = sphere.from_line(&l);
	// 			let roots = calc_mobius_everywhere(&ray, 30);
	// 			match roots {
	// 				Roots::Zero => (128, 128, 128),
	// 				Roots::One(..) => (0, 0, 128),
	// 				Roots::Two(..) => (128, 0, 0),
	// 			}
	// 		})
	// 		.unwrap_or((0, 0, 0));
	// 	image.set_pixel(x, y, color);
	// }

	// image.save("a.png");

	let count = 40;

	let points = MobiusPoints::calc(count);
	// points.save();

	// let points = MobiusPoints::load();

	// let mut visited_points = BTreeMap::new();

	let g = colorgrad::turbo();

	// let min = points
	// 	.solved_points
	// 	.iter()
	// 	.filter_map(|(_, should_be)| match should_be {
	// 		Roots::Zero => None,
	// 		Roots::One(a) => Some(a.0),
	// 		Roots::Two(a, b) => Some(if a.0 < b.0 { a.0 } else { b.0 }),
	// 	})
	// 	.min_by(|a, b| a.partial_cmp(b).unwrap())
	// 	.unwrap();
	// let max = points
	// 	.solved_points
	// 	.iter()
	// 	.filter_map(|(_, should_be)| match should_be {
	// 		Roots::Zero => None,
	// 		Roots::One(a) => Some(a.0),
	// 		Roots::Two(a, b) => Some(if a.0 > b.0 { a.0 } else { b.0 }),
	// 	})
	// 	.max_by(|a, b| a.partial_cmp(b).unwrap())
	// 	.unwrap();

	// dbg!(min, max);

	let mut image = Image::new(count * count, count * count);
	for (line, should_be) in points.solved_points {
		let alpha1 = line.o.alpha.0 / (2. * PI);
		let beta1 = line.o.beta.0 / PI;
		let alpha2 = line.od.alpha.0 / (2. * PI);
		let beta2 = line.od.beta.0 / PI;

		assert_between!(0., alpha1, 1.);
		assert_between!(0., beta1, 1.);
		assert_between!(0., alpha2, 1.);
		assert_between!(0., beta2, 1.);

		let alpha1i = (alpha1 * count as f64).round() as usize;
		let beta1i = (beta1 * count as f64).round() as usize;
		let alpha2i = (alpha2 * count as f64).round() as usize;
		let beta2i = (beta2 * count as f64).round() as usize;

		assert_between!(0, alpha1i, count);
		assert_between!(0, beta1i, count);
		assert_between!(0, alpha2i, count);
		assert_between!(0, beta2i, count);

		let x = alpha1i * count + alpha2i;
		let y = beta1i * count + beta2i;


		let color = match should_be {
			Roots::Zero => None,
			Roots::One(a) => Some(a.0),
			Roots::Two(a, b) => Some(if a.0 < b.0 { a.0 } else { b.0 }),
		};
		let color = if let Some(mut color) = color {
			// if color < 0.0001 {
			// 	dbg!(alpha1, beta1, alpha2, beta2, alpha1i, beta1i, alpha2i, beta2i, x, y, color);
			// 	panic!();
			// }

			// assert_between!(min - 0.01, color, max + 0.01);
			// color = (color - min + (max - min) * 0.05) / (max - min) / 1.05;
			let (r, g, b, _) = g.at(color.into()).to_lrgba_u8();
			// let r = color * 256 * 256 * 256
			(r, g, b)
		} else {
			(0, 0, 0)
		};

		image.set_pixel(x, y, color);

		// if let Some(v) = visited_points.get(&(x, y)) {
		// 	dbg!(alpha1, beta1, alpha2, beta2, alpha1i, beta1i, alpha2i, beta2i, x, y, color, should_be, v);
		// 	panic!();
		// }
		// visited_points.insert((x, y), (alpha1, beta1, alpha2, beta2));
	}
	image.save("texture.png");
}
