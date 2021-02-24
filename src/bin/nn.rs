/*use argmin::{
	prelude::*,
	solver::{
		conjugategradient::{NonlinearConjugateGradient, PolakRibiere},
		gradientdescent::SteepestDescent,
		linesearch::*,
		quasinewton::{SR1TrustRegion, BFGS, LBFGS},
		trustregion::Steihaug,
	},
};
use finitediff::*;
use glam::{Mat4, Vec4};
use ndarray::{Array1, Array2};
#[allow(unused_imports)]
use openblas_src::*;
use portals::{Intersections, MobiusPoints};
use rand::prelude::*;

struct Mobius {
	data: MobiusPoints,
}

fn relu(input: &mut f32) {
	if *input < 0. {
		*input = 0.;
	}
}

fn sigmoid(input: &mut f32) {
	let ex = input.exp();
	*input = ex / (ex + 1.);
}

fn vec_to_intersections(x: Vec4) -> Intersections {
	if x.x < 0. && x.z < 0. {
		Intersections::None
	} else if x.x > 0. && x.z < 0. {
		Intersections::One { u: x.x, t: x.y }
	} else if x.z > 0. && x.x < 0. {
		Intersections::One { u: x.z, t: x.w }
	} else {
		Intersections::Two { u1: x.x, t1: x.y, u2: x.z, t2: x.w }
	}
}

fn cost(data: &MobiusPoints, p: &[f32]) -> f64 {
	let matrices = p.len() / 16;
	let mut sum: f64 = 0.;

	for (input, result) in data {
		let mut x: Vec4 = input.clone().into();
		for matrix in 0..matrices {
			if matrix != 0 {
				relu(&mut x.x);
				relu(&mut x.y);
				relu(&mut x.z);
				relu(&mut x.w);
			}
			let arr: [f32; 16] = [
				p[matrix * 16 + 0] as f32,
				p[matrix * 16 + 1] as f32,
				p[matrix * 16 + 2] as f32,
				p[matrix * 16 + 3] as f32,
				p[matrix * 16 + 4] as f32,
				p[matrix * 16 + 5] as f32,
				p[matrix * 16 + 6] as f32,
				p[matrix * 16 + 7] as f32,
				p[matrix * 16 + 8] as f32,
				p[matrix * 16 + 9] as f32,
				p[matrix * 16 + 10] as f32,
				p[matrix * 16 + 11] as f32,
				p[matrix * 16 + 12] as f32,
				p[matrix * 16 + 13] as f32,
				p[matrix * 16 + 14] as f32,
				p[matrix * 16 + 15] as f32,
			];
			x = Mat4::from_cols_array(&arr) * x;
		}

		let actual = vec_to_intersections(x);
		sum += distance(result.clone(), actual) as f64;
	}

	sum / (data.len() as f64)
}

impl ArgminOp for Mobius {
	type Float = f64;
	type Hessian = Array2<f64>;
	type Jacobian = ();
	type Output = f64;
	type Param = Array1<f64>;

	fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
		Ok(cost(&self.data, &p.to_vec().into_iter().map(|x| x as f32).collect::<Vec<_>>()))
	}

	fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
		Ok((*p).forward_diff(&|x| self.apply(&x).unwrap()))
	}

	fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
		Ok((*p).forward_hessian(&|x| self.gradient(&x).unwrap()))
	}
}

fn run(matrices: usize, data: MobiusPoints, init: &[f32]) -> Result<(), Error> {
	let cost = Mobius { data };

	// Define initial parameter vector
	let init_param: Array1<f64> = init.iter().map(|x| *x as f64).collect::<Vec<_>>().into();
	let init_hessian: Array2<f64> = Array2::eye(matrices * 16);

	let linesearch = MoreThuenteLineSearch::new().c(1e-4, 0.9)?;
	// let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.5)?).rho(0.9)?;
	// let linesearch = HagerZhangLineSearch::new();

	// let solver = BFGS::new(init_hessian, linesearch);
	// let solver = NonlinearConjugateGradient::new(linesearch, PolakRibiere::new())?.restart_iters(10).restart_orthogonality(0.1);
	// let solver = SteepestDescent::new(linesearch);
	let solver = LBFGS::new(linesearch, 7);

	// let solver = SR1TrustRegion::new(Steihaug::new().max_iters(20));

	// Run solver
	let res = Executor::new(cost, solver, init_param)
		.add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
		.max_iters(10000)
		.run()?;

	// Wait a second (lets the logger flush everything before printing again)
	std::thread::sleep(std::time::Duration::from_secs(1));

	// Print result
	println!("{}", res);
	Ok(())
}

use differential_evolution::self_adaptive_de;

fn main() {
	let data1 = &data;

	let matrices = 4;
	let mut de = self_adaptive_de(vec![(-1.0, 1.0); matrices * 16], |pos| cost(data1, pos) as f32);
	de.iter().nth(1000);

	// show the result
	let (cost, pos) = de.best().unwrap();
	println!("cost: {}", cost);
	println!("pos: {:?}", pos);

	if let Err(ref e) = run(matrices, data.clone(), pos) {
		println!("{}", e);
		std::process::exit(1);
	}
}
*/
