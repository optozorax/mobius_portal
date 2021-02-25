pub fn mymin(a: f64, b: f64) -> f64 { if a < b { a } else { b } }

pub mod option {
	pub trait AnyOrBothWith {
		type Inner;
		fn any_or_both_with<F: FnOnce(Self::Inner, Self::Inner) -> Self::Inner>(
			self,
			b: Option<Self::Inner>,
			f: F,
		) -> Option<Self::Inner>;
	}

	impl<T> AnyOrBothWith for Option<T> {
		type Inner = T;

		fn any_or_both_with<F: FnOnce(Self::Inner, Self::Inner) -> Self::Inner>(
			self,
			b: Option<Self::Inner>,
			f: F,
		) -> Option<Self::Inner> {
			match (self, b) {
				(Some(a), Some(b)) => Some((f)(a, b)),
				(Some(a), None) => Some(a),
				(None, Some(b)) => Some(b),
				(None, None) => None,
			}
		}
	}
}

#[macro_use]
pub mod assert {
	#[macro_export]
	macro_rules! assert_between {
		($lt:expr, $val:expr, $gt:expr) => {{
			if $val < $lt {
				panic!("{} < {} violated with values {} < {}", stringify!($lt), stringify!($val), $lt, $val);
			}
			if $val >= $gt {
				panic!("{} < {} violated with values {} < {}", stringify!($val), stringify!($gt), $val, $gt);
			}
		}};
	}
}

pub mod numeric_methods {
	pub trait FloatFunction {
		fn calc(&self, x: f64) -> f64;
	}

	pub struct Derivative<F>(pub F);

	impl<F: FloatFunction> FloatFunction for Derivative<F> {
		fn calc(&self, x: f64) -> f64 {
			const H: f64 = 0.001;
			(self.0.calc(x + 2. * H) + 8. * self.0.calc(x + H) - 8. * self.0.calc(x - H) - self.0.calc(x - 2. * H))
				/ (12. * H)
		}
	}

	pub fn newton_1d<F, DF, XF>(f: &F, df: &DF, xf: XF, init: f64, max_iter: i32, eps: f64) -> Option<f64>
	where
		F: FloatFunction,
		DF: FloatFunction,
		XF: Fn(f64, f64) -> f64,
	{
		let mut x = init;
		for _ in 0..max_iter {
			let fx = f.calc(x);
			if fx < eps {
				return Some(x);
			}
			let dx = -fx / df.calc(x);
			x = xf(x, dx);
		}
		None
	}
}

pub mod line {
	use glam::{DMat3 as Mat3, DVec3 as Vec3};

	pub struct Line1 {
		pub o: f64,
		pub d: f64,
	}

	impl Line1 {
		pub fn from(&self, t: LinePosition) -> f64 { self.o + self.d * t.0 }

		pub fn to(&self, o: f64) -> LinePosition { LinePosition((o - self.o) / self.d) }
	}

	#[derive(Debug, Clone)]
	pub struct Line3 {
		pub o: Vec3,
		pub d: Vec3,
	}

	#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
	pub struct LinePosition(pub f64);

	impl Line3 {
		pub fn from(&self, t: LinePosition) -> Vec3 { self.o + self.d * t.0 }

		pub fn from_dir(&self, t: LinePosition) -> Vec3 { self.d * t.0 }

		pub fn to(&self, o: Vec3) -> LinePosition { LinePosition((o - self.o).dot(self.d)) }

		pub fn to_dir(&self, d: Vec3) -> LinePosition { LinePosition(d.dot(self.d)) }

		pub fn distance(&self, o: Vec3) -> f64 { (self.from(self.to(o)) - o).length() }

		pub fn distance_skew_line(&self, other: &Line3) -> f64 {
			let v = self.d.cross(other.d);
			(self.o - other.d).dot(v).abs() / v.length()
		}

		pub fn nearest_points_to_skew_line(&self, other: &Line3) -> (LinePosition, LinePosition) {
			let pos = Mat3::from_cols(self.d, other.d, self.d.cross(other.d)).inverse() * (other.o - self.o);
			(LinePosition(pos.x), LinePosition(-pos.y))
		}

		pub fn apply(&self, line1: Line1) -> Line3 {
			Line3 { o: self.from(LinePosition(line1.o)), d: self.from_dir(LinePosition(line1.d)) }
		}
	}

	#[cfg(test)]
	mod tests {
		use super::*;

		#[test]
		fn line() {
			let a = Line3 { o: Vec3::new(5., 0., 0.), d: Vec3::new(0., 0., 1.) };
			let b = Line3 { o: Vec3::new(0., 0., 4.), d: Vec3::new(1., 0., 0.) };
			let (t_a, t_b) = a.nearest_points_to_skew_line(&b);
			approx::assert_relative_eq!(t_a.0, 4.);
			approx::assert_relative_eq!(t_b.0, 5.);
		}
	}
}

pub mod angle {
	use std::f64::consts::PI;

	use serde::{Deserialize, Serialize};

	#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Serialize, Deserialize)]
	pub struct Degrees(pub f64);

	#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Serialize, Deserialize)]
	pub struct Radians(pub f64);

	impl From<Degrees> for Radians {
		fn from(degrees: Degrees) -> Radians { Radians(degrees.0 / 180. * PI) }
	}

	impl From<Radians> for Degrees {
		fn from(radians: Radians) -> Degrees { Degrees(radians.0 / PI * 180.) }
	}

	pub fn clamp_mod(a: f64, max: f64) -> f64 { (max + a % max) % max }

	pub fn clamp_angle(angle: Radians) -> Radians { Radians(clamp_mod(angle.0, 2. * PI)) }

	#[derive(Clone, Debug)]
	pub struct RadiansIter {
		to: Radians,
		count: usize,
		current: usize,
	}

	pub fn iter_2pi(count: usize) -> RadiansIter { RadiansIter { to: Radians(2. * PI), count, current: 0 } }

	pub fn iter_pi(count: usize) -> RadiansIter { RadiansIter { to: Radians(PI), count, current: 0 } }

	impl Iterator for RadiansIter {
		type Item = Radians;

		fn next(&mut self) -> Option<Self::Item> {
			if self.current == self.count {
				None
			} else {
				let mut output = self.to;
				output.0 *= self.current as f64 / self.count as f64;
				self.current += 1;
				Some(output)
			}
		}
	}

	impl ExactSizeIterator for RadiansIter {
		fn len(&self) -> usize { self.count }
	}
}

pub mod quadratic_equation {
	use serde::{Deserialize, Serialize};

	#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
	pub enum Roots<T> {
		Two(T, T),
		One(T),
		Zero,
	}

	impl<T> Roots<T> {
		pub fn map<F: Fn(T) -> Y, Y>(self, f: F) -> Roots<Y> {
			use Roots::*;
			match self {
				Two(a, b) => Two(f(a), f(b)),
				One(a) => One(f(a)),
				Zero => Zero,
			}
		}

		pub fn filter_map<F: Fn(T) -> Option<Y>, Y>(self, f: F) -> Roots<Y> {
			use Roots::*;
			match self {
				Two(a, b) => {
					if let Some(a) = f(a) {
						if let Some(b) = f(b) { Two(a, b) } else { One(a) }
					} else {
						if let Some(b) = f(b) { One(b) } else { Zero }
					}
				},
				One(a) => {
					if let Some(a) = f(a) {
						One(a)
					} else {
						Zero
					}
				},
				Zero => Zero,
			}
		}
	}

	pub fn solve_quadratic_equation(a: f64, b: f64, c: f64) -> Roots<f64> {
		use Roots::*;
		let d = b * b - 4.0 * a * c;
		if d < 0.0 {
			Zero
		} else if d.abs() < 1e-9 {
			One(-b / (2.0 * a))
		} else {
			let sq_d = d.sqrt();
			let x1 = (-b + sq_d) / (2.0 * a);
			let x2 = (-b - sq_d) / (2.0 * a);
			Two(x1.max(x2), x1.min(x2))
		}
	}
}

pub mod sphere {
	use std::f64::consts::PI;

	use glam::DVec3 as Vec3;
	use serde::{Deserialize, Serialize};

	use crate::{angle::Radians, line::*, quadratic_equation::*};

	#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
	pub struct Angles3 {
		pub alpha: Radians,
		pub beta: Radians,
	}

	impl Angles3 {
		pub fn new(alpha: Radians, beta: Radians) -> Self { Angles3 { alpha, beta } }
	}

	#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
	pub struct Sph3 {
		pub angles: Angles3,
		pub r: f64,
	}

	impl Sph3 {
		pub fn new(alpha: Radians, beta: Radians, r: f64) -> Self { Sph3 { angles: Angles3::new(alpha, beta), r } }
	}

	impl From<Vec3> for Angles3 {
		fn from(o: Vec3) -> Angles3 {
			Angles3 { alpha: Radians(o.z.atan2(o.x) + PI), beta: Radians((o.x * o.x + o.z * o.z).sqrt().atan2(o.y)) }
		}
	}

	impl From<Vec3> for Sph3 {
		fn from(o: Vec3) -> Sph3 { Sph3 { angles: o.into(), r: (o.x * o.x + o.y * o.y + o.z * o.z).sqrt() } }
	}

	impl From<Angles3> for Vec3 {
		fn from(s: Angles3) -> Vec3 {
			Vec3::new(-s.beta.0.sin() * s.alpha.0.cos(), s.beta.0.cos(), -s.beta.0.sin() * s.alpha.0.sin())
		}
	}

	impl From<Sph3> for Vec3 {
		fn from(s: Sph3) -> Vec3 { Vec3::from(s.angles) * s.r }
	}

	#[derive(Debug, Clone, Serialize, Deserialize)]
	pub struct Sphere3 {
		pub pos: Vec3,
		pub r: f64,
	}

	#[derive(Debug, Clone, Serialize, Deserialize)]
	pub struct SphereLine3 {
		pub o: Angles3,
		pub od: Angles3,
	}

	impl Sphere3 {
		pub fn intersect_line(&self, line: &Line3) -> Roots<Angles3> {
			let p = line.o - self.pos;

			let a = line.d.dot(line.d);
			let b = 2.0 * p.dot(line.d);
			let c = p.dot(p) - self.r * self.r;

			solve_quadratic_equation(a, b, c).map(|t| self.to(line.from(LinePosition(t))))
		}

		pub fn from(&self, pos: Angles3) -> Vec3 { self.pos + Vec3::from(pos) * self.r }

		pub fn to(&self, pos: Vec3) -> Angles3 { Angles3::from(pos - self.pos) }

		pub fn from_line(&self, line: &SphereLine3) -> Line3 {
			let o = self.from(line.o.clone());
			let d = self.from(line.od.clone());
			Line3 { o, d: d - o }
		}

		pub fn to_line(&self, line: &Line3) -> Option<(SphereLine3, Line1)> {
			match self.intersect_line(line) {
				Roots::Two(a, b) => {
					let al = line.to(self.from(a.clone()));
					let bl = line.to(self.from(b.clone()));
					Some((SphereLine3 { o: a, od: b }, Line1 { o: al.0, d: bl.0 - al.0 }))
				},
				_ => None,
			}
		}
	}

	#[cfg(test)]
	mod tests {
		use glam::DVec3 as Vec3;

		use super::*;

		#[test]
		fn angles() {
			fn test_vec3(input: Vec3) {
				let sph3 = Sph3::from(input);
				let new_input = Vec3::from(sph3);
				if !approx::abs_diff_eq!((new_input - input).length(), 0., epsilon = 0.01) {
					dbg!(input, sph3, new_input);
					panic!();
				}
				assert!(sph3.r >= 0.);
				assert_between!(0., sph3.angles.alpha.0, 2. * PI);
				assert_between!(0., sph3.angles.beta.0, PI);
			}

			test_vec3(Vec3::new(1., 0., 0.));
			test_vec3(Vec3::new(0., 1., 0.));
			test_vec3(Vec3::new(0., 0., 1.));
			test_vec3(Vec3::new(1., 1., 0.));
			test_vec3(Vec3::new(0., 1., 1.));
			test_vec3(Vec3::new(1., 0., 1.));
			test_vec3(Vec3::new(1., 1., 1.));

			test_vec3(Vec3::new(20., 3., 10.));
			test_vec3(Vec3::new(-20., 3., 10.));
			test_vec3(Vec3::new(20., -3., 10.));
			test_vec3(Vec3::new(20., 3., -10.));
			test_vec3(Vec3::new(-20., -3., 10.));
			test_vec3(Vec3::new(20., -3., -10.));
			test_vec3(Vec3::new(-20., 3., -10.));
			test_vec3(Vec3::new(-20., -3., -10.));

			test_vec3(Vec3::new(11., -103., 0.));
		}
	}
}

pub mod image {
	use std::{fs::File, io::BufWriter, path::Path};

	use glam::DVec2 as Vec2;

	pub struct ImageIterator {
		x: usize,
		y: usize,
		w: usize,
		h: usize,
	}

	impl Iterator for ImageIterator {
		type Item = (usize, usize, Vec2);

		fn next(&mut self) -> Option<Self::Item> {
			if self.y == self.h {
				return None;
			}

			let min = std::cmp::min(self.w, self.h) as f64;
			let to_return = (self.x, self.y, (Vec2::new(self.x as f64, self.y as f64) / min * 2. - Vec2::new(1., 1.)));

			self.x += 1;
			if self.x == self.w {
				self.y += 1;
				self.x = 0;
			}
			Some(to_return)
		}
	}

	pub struct Image {
		w: usize,
		h: usize,
		data: Vec<u8>,
	}

	impl Image {
		pub fn new(w: usize, h: usize) -> Self { Self { w, h, data: vec![0; w * h * 3] } }

		pub fn iter(&self) -> ImageIterator { ImageIterator { x: 0, y: 0, w: self.w, h: self.h } }

		pub fn set_pixel(&mut self, x: usize, y: usize, color: (u8, u8, u8)) {
			let offset = (x + y * self.w) * 3;
			self.data[offset + 0] = color.0;
			self.data[offset + 1] = color.1;
			self.data[offset + 2] = color.2;
		}

		pub fn save(&self, filename: &str) {
			let path = Path::new(filename);
			let file = File::create(path).unwrap();
			let ref mut wr = BufWriter::new(file);

			let mut encoder = png::Encoder::new(wr, self.w as u32, self.h as u32);
			encoder.set_color(png::ColorType::RGB);
			encoder.set_depth(png::BitDepth::Eight);
			let mut writer = encoder.write_header().unwrap();

			writer.write_image_data(&self.data).unwrap();
		}
	}
}

pub mod cam {
	use glam::{DMat3 as Mat3, DVec2 as Vec2, DVec3 as Vec3};

	use crate::{angle::Radians, line::*};

	pub struct Cam {
		crd: Mat3,
		pos: Vec3,
		view: f64,
	}

	impl Cam {
		pub fn look_at_from(look_at: Vec3, from: Vec3, view_angle: Radians) -> Cam {
			let view = (view_angle.0 / 2.).tan();

			let up = Vec3::new(0., 1., 0.);

			let k = (look_at - from).normalize();
			let i = up.cross(k).normalize();
			let j = k.cross(i).normalize();

			Cam { crd: Mat3::from_cols(i, j, k), pos: from, view }
		}

		/// Pos should be in [-1; 1]²
		pub fn get_ray(&self, pos: Vec2) -> Line3 {
			Line3 { o: self.pos, d: (self.crd * Vec3::new(pos.x * self.view, pos.y * self.view, 1.)).normalize() }
		}
	}
}

pub mod mobius {
	use std::{f64::consts::PI, io::Write};

	use glam::DVec3 as Vec3;
	use indicatif::{ProgressBar, ProgressStyle};
	use serde::{Deserialize, Serialize};

	use crate::{angle::*, line::*, mymin, numeric_methods::*, quadratic_equation::Roots, sphere::*};

	pub fn mobius_ray(u: Radians) -> Line3 {
		let u = u.0;
		let u2 = u / 2.;
		Line3 {
			o: Vec3::new(u.cos(), 0., u.sin()),
			d: Vec3::new(u2.cos() * u.cos(), u2.sin(), u2.cos() * u.sin()) / 2.,
		}
	}

	#[derive(Debug, Clone)]
	pub struct ApproachMobiusResult {
		pub distance: f64,
		pub cost: f64,
		pub t_line: LinePosition,
		pub t_mobius: LinePosition,
	}

	pub fn approach_mobius(u: Radians, ray: &Line3) -> ApproachMobiusResult {
		let mray = mobius_ray(u);

		let (t_line, t_mobius) = ray.nearest_points_to_skew_line(&mray);

		let up = mray.from(LinePosition(1.));
		let down = mray.from(LinePosition(-1.));

		let t_up = ray.to(up);
		let t_down = ray.to(down);

		let distance = if t_mobius.0.abs() < 1. {
			(mray.from(t_mobius) - ray.from(t_line)).length()
		} else {
			mymin((up - ray.from(t_up)).length(), (down - ray.from(t_down)).length())
		};

		let cost = {
			let result = distance;
			// if t_mobius.0.abs() > 1. {
			// 	result += 2. * t_mobius.0.abs();
			// }
			// if t_line.0 < 0. {
			// 	result *= 4. * (1. + t_line.0.abs());
			// };
			result
		};

		ApproachMobiusResult { distance, cost, t_mobius, t_line }
	}

	pub struct Approx {
		pub best_approach: ApproachMobiusResult,
		pub best_u: Radians,
	}

	// pub fn update_best_approx(best: Option<Approx>, new: Option<Approx>) -> Option<Approx> {
	// 	best.any_or_both_with(new, |b, n| if n.best_approach.cost < b.best_approach.cost { n } else { b })
	// }

	pub fn calc_mobius_from(ray: &Line3, u: Radians) -> Option<Approx> {
		#[derive(Debug, Clone)]
		pub struct RayToMobius<'a> {
			pub ray: &'a Line3,
		}

		impl FloatFunction for RayToMobius<'_> {
			fn calc(&self, u: f64) -> f64 { approach_mobius(Radians(u), self.ray).cost }
		}

		newton_1d(
			&RayToMobius { ray },
			&Derivative(RayToMobius { ray }),
			|x, dx| clamp_angle(Radians(x + dx)).0,
			u.0,
			30,
			5e-4,
		)
		.map(|u| Radians(u))
		.map(|best_u| Approx { best_approach: approach_mobius(best_u, &ray), best_u })
	}

	pub fn calc_mobius_everywhere(ray: &Line3, count: usize) -> Roots<Radians> {
		let mut roots: Vec<(Radians, f64)> = Vec::new();
		let mut add_root = |new_root: Radians, distance: f64| {
			if let Some(pos) = roots.iter().position(|r| (r.0.0 - new_root.0).abs() / (2. * PI) < 7e-3) {
				if roots[pos].1 > distance {
					roots[pos] = (new_root, distance);
				}
			} else {
				roots.push((new_root, distance));
			}
		};

		for u in iter_2pi(count) {
			// let approx = approach_mobius(u, ray);
			// if approx.distance * (count as f64) < 2. && approx.t_mobius.0.abs() < 1. {
			// 	assert_between!(0., approx.t_line.0, 1.);
			// 	add_root(u);
			// }
			if let Some(approx) = calc_mobius_from(ray, u) {
				if approx.best_approach.t_mobius.0.abs() < 1. {
					assert_between!(0., approx.best_approach.t_line.0, 1.);
					add_root(approx.best_u, approx.best_approach.distance);
				}
			}
		}

		match roots[..] {
			[] => Roots::Zero,
			[a] => Roots::One(a.0),
			[a, b] => Roots::Two(a.0, b.0),
			[a, b, ..] => Roots::Two(a.0, b.0),
			_ => panic!(
				"Mobius band must has maximum two intersections with ray. This violated with u's = {:#?} and ray = {:?}",
				roots, ray
			),
		}
	}

	#[derive(Debug, Clone, Serialize, Deserialize)]
	pub struct MobiusPoints {
		pub sphere: Sphere3,
		pub solved_points: Vec<(SphereLine3, Roots<Radians>)>,
	}

	pub fn roots_distance<T: Copy, F: Fn(T, T) -> f64>(
		should_be: Roots<T>,
		actual: Roots<T>,
		f: F,
		extra_root_penalty: f64,
	) -> f64 {
		match should_be {
			Roots::Zero => match actual {
				Roots::Zero => 0.,
				Roots::One(..) => extra_root_penalty,
				Roots::Two(..) => 2. * extra_root_penalty,
			},
			Roots::One(a) => match actual {
				Roots::Zero => extra_root_penalty,
				Roots::One(a1) => f(a, a1),
				Roots::Two(a1, b1) => extra_root_penalty + mymin(f(a, a1), f(a, b1)),
			},
			Roots::Two(a, b) => match actual {
				Roots::Zero => 2. * extra_root_penalty,
				Roots::One(a1) => extra_root_penalty + mymin(f(a, a1), f(b, a1)),
				Roots::Two(a1, b1) => mymin(f(a, a1) + f(b, b1), f(a, b1) + f(b, a1)),
			},
		}
	}

	impl MobiusPoints {
		pub fn sphere() -> Sphere3 {
			// Mobius strip is fully covered by this sphere
			Sphere3 { pos: Vec3::new(0., 0., 0.), r: 1.55 }
		}

		pub fn calc(count: usize) -> Self {
			fn intersect_cylinder(ray: &Line3) -> Roots<Radians> {
				let v = ray.d;
				let p = ray.o;
				let d = Vec3::new(0., 1., 0.);
				let r = 0.5;

				let vv = v.dot(v);
				let pp = p.dot(p);
				let dd = d.dot(d);

				let dv = d.dot(v);
				let dp = d.dot(p);
				let pv = p.dot(v);

				let a = dd * vv - dv * dv;
				let b = 2. * (dd * pv - dp * dv);
				let c = dd * pp - dp * dp - r * r * dd;

				crate::quadratic_equation::solve_quadratic_equation(a, b, c).filter_map(|t| {
					Some(t)
						.filter(|t| {
							let pos = ray.from(LinePosition(*t));
							pos.y.abs() < 0.4
						})
						.map(|t| Radians(t))
				})
			}

			let sphere = Self::sphere();
			let mut solved_points = Vec::new();
			let bar =
				ProgressBar::new((count * count) as u64).with_style(ProgressStyle::default_bar().template(
					"[elapsed: {elapsed_precise:>8} | remaining: {eta_precise:>8} | {percent:>3}%] {wide_bar}",
				));
			for alpha1 in iter_2pi(count) {
				for beta1 in iter_pi(count) {
					bar.inc(1);
					bar.tick();
					std::io::stdout().flush().unwrap();
					let o = Angles3::new(alpha1, beta1);
					for alpha2 in iter_2pi(count) {
						for beta2 in iter_pi(count) {
							let od = Angles3::new(alpha2, beta2);
							let sphere_line = SphereLine3 { o, od };
							let ray = sphere.from_line(&sphere_line);
							let roots = calc_mobius_everywhere(&ray, 30);
							// let roots = intersect_cylinder(&ray);
							solved_points.push((sphere_line, roots));
						}
					}
				}
			}
			bar.finish();
			Self { sphere, solved_points }
		}

		pub fn load() -> Self {
			let file = std::fs::read("points.bin").unwrap();
			bincode::deserialize(&file[..]).unwrap()
		}

		pub fn save(&self) {
			use std::{fs::File, io::prelude::*};
			let mut file = File::create("points.bin").unwrap();
			file.write_all(&bincode::serialize(&self).unwrap()).unwrap();
		}

		pub fn distance<F: Fn(&SphereLine3) -> Roots<Radians>>(&self, f: F) -> f64 {
			let mut sum: f64 = 0.;
			for (line, should_be) in &self.solved_points {
				let actual = f(line);
				sum += roots_distance(*should_be, actual, |a, b| (a.0 - b.0).abs(), 30.) as f64;
			}
			sum as f64
		}
	}
}
