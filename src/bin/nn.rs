use std::f64::consts::PI;

use argmin::{
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
use glam::{DMat4 as Mat4, DVec4 as Vec4};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array2};
#[allow(unused_imports)]
use openblas_src::*;
use portals::{
	angle::Radians,
	mobius::{calc_mobius_from, random_sphere_line, update_best_approx, ApproachMobiusResult, Approx, MobiusPoints},
	sphere::SphereLine3,
};
use rand::prelude::*;

struct Mobius {
	data: MobiusPoints,
}

fn relu(input: &mut f64) {
	if *input < 0. {
		*input = 0.;
	}
}

fn elu(input: &mut f64) {
	if *input < 0. {
		*input = input.exp() - 1.;
	}
}

fn elu_vec(vec: &mut Vec4) {
	elu(&mut vec.x);
	elu(&mut vec.y);
	elu(&mut vec.z);
	elu(&mut vec.w);
}

fn relu_vec(vec: &mut Vec4) {
	relu(&mut vec.x);
	relu(&mut vec.y);
	relu(&mut vec.z);
	relu(&mut vec.w);
}

fn sigmoid(input: &mut f64) {
	let ex = input.exp();
	*input = ex / (ex + 1.);
}

fn non_lineriality(vec: &mut Vec4) {
	relu_vec(vec)

	// sigmoid(&mut vec.x);
	// sigmoid(&mut vec.y);
	// sigmoid(&mut vec.z);
	// sigmoid(&mut vec.w);
}

fn cost(data: &MobiusPoints, p: &[f64]) -> f64 {
	let matrices = (0..p.len() / 20)
		.map(|matrix| matrix * 20)
		.map(|offset| {
			#[rustfmt::skip]
			let arr: [f64; 16] = [
				p[offset + 0], p[offset + 1], p[offset + 2], p[offset + 3],
				p[offset + 4], p[offset + 5], p[offset + 6], p[offset + 7],
				p[offset + 8], p[offset + 9], p[offset + 10], p[offset + 11],
				p[offset + 12], p[offset + 13], p[offset + 14], p[offset + 15],
			];
			(
				Mat4::from_cols_array(&arr),
				Vec4::new(p[offset + 16], p[offset + 17], p[offset + 18], p[offset + 19]),
			)
		})
		.collect::<Vec<(Mat4, Vec4)>>();
	let f = |line: &SphereLine3| {
		let mut vec = Vec4::new(line.o.alpha.0, line.o.beta.0, line.od.alpha.0, line.od.beta.0);
		for (pos, (matrix, bias)) in matrices.iter().enumerate() {
			if pos != 0 {
				non_lineriality(&mut vec);
			}
			vec = *matrix * vec + *bias;
		}
		if vec.x < 0. {
			None
		} else {
			let sphere = MobiusPoints::sphere();
			let ray = sphere.from_line(line);
			let mut best: Option<Approx> = None;
			if let Some(approx) = calc_mobius_from(&ray, Radians(vec.x % (2. * PI))) {
				if approx.best_approach.t_mobius.0.abs() < 1. && approx.best_approach.distance < 5e-4 {
					best = update_best_approx(Some(approx), best);
				}
			}
			best.map(|approx| approx.best_u)
		}
	};
	data.distance(f)
	// data.distance_by_random(&mut rand::thread_rng(), f)
}

impl ArgminOp for Mobius {
	type Float = f64;
	type Hessian = Array2<f64>;
	type Jacobian = ();
	type Output = f64;
	type Param = Array1<f64>;

	fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
		Ok(cost(&self.data, &p.to_vec().into_iter().collect::<Vec<_>>()))
	}

	fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
		Ok((*p).forward_diff(&|x| self.apply(&x).unwrap()))
	}

	fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
		Ok((*p).forward_hessian(&|x| self.gradient(&x).unwrap()))
	}
}

fn run(matrices: usize, data: MobiusPoints, init: &[f64]) -> Result<(), Error> {
	let cost = Mobius { data };
	let init_param: Array1<f64> = init.iter().map(|x| *x as f64).collect::<Vec<_>>().into();
	let init_hessian: Array2<f64> = Array2::eye(matrices * 20);

	let linesearch = {
		MoreThuenteLineSearch::new().c(1e-4, 0.9)?
		// BacktrackingLineSearch::new(ArmijoCondition::new(0.5)?).rho(0.9)?
		// HagerZhangLineSearch::new()
	};

	let solver = {
		// BFGS::new(init_hessian, linesearch)
		// NonlinearConjugateGradient::new(linesearch, PolakRibiere::new())?
		// .restart_iters(10)
		// .restart_orthogonality(0.1)
		// SteepestDescent::new(linesearch)
		LBFGS::new(linesearch, 7)
	};

	// let solver = SR1TrustRegion::new(Steihaug::new().max_iters(20));

	let res = Executor::new(cost, solver, init_param)
		.add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
		.max_iters(10000)
		.run()?;

	std::thread::sleep(std::time::Duration::from_secs(1));

	println!("{}", res);
	Ok(())
}

use std::time::{Duration, Instant};

use differential_evolution::self_adaptive_de;

pub fn time<F: FnOnce()>(f: F) -> Duration {
	let now = Instant::now();
	f();
	now.elapsed()
}

struct FromKeras {
	matrices: Vec<Mat4>,
	biases: Vec<Vec4>,
	last: Vec4,
	last_val: f64,
}

impl FromKeras {
	fn new() -> Self {
		Self {
			matrices: vec![
				Mat4::from_cols_array_2d(&[
					[1.0165445, -0.3549156, 0.06285402, -0.67766607],
					[0.14903155, -1.8064088, -0.38887787, 0.16576281],
					[-1.6730295, -0.01575787, -0.6652524, 0.14698108],
					[-0.11340991, -0.5738989, -0.17948157, 0.07672196],
				]),
				Mat4::from_cols_array_2d(&[
					[-0.61962205, 0.35224357, -0.7050133, -1.007992],
					[-1.9364115, 0.20592041, 1.4574093, -0.35939673],
					[1.4148147, 0.6364147, -3.2876616, 2.146241],
					[-2.1854484, 1.271743, -1.0198306, 0.08767886],
				]),
				Mat4::from_cols_array_2d(&[
					[-0.56710607, 3.0085683, -1.7664146, -0.84609425],
					[1.0190496, -0.20223191, 0.7935057, -0.19631511],
					[-0.4405343, 3.1800041, -2.201156, -0.6110492],
					[-2.0771947, 1.9290844, 4.0726914, -2.9167714],
				]),
				Mat4::from_cols_array_2d(&[
					[-0.8140761, 0.38469988, 0.42777032, -1.3800956],
					[-3.0025988, 0.3517017, 0.16559923, -2.7702587],
					[0.85082966, 0.693335, -0.5511722, 0.41399848],
					[0.69392604, 0.5646706, -2.344027, 0.4209389],
				]),
			],
			biases: vec![
				Vec4::new(1.0014485, 4.026909, 2.5342689, 1.7052656),
				Vec4::new(0.6362909, 0.22164212, -0.09167471, 0.51322347),
				Vec4::new(-0.38457462, -2.7439218, -2.7083, 0.58849967),
				Vec4::new(-0.66961056, 0.89743716, -1.3223268, 1.3417816),
			],
			last: Vec4::new(-1.8168594, 0.81896317, -0.6204357, 1.971852),
			last_val: 0.8696187,
		}
	}

	fn calc(&self, mut input: Vec4) -> f64 {
		for (matrix, bias) in self.matrices.iter().zip(self.biases.iter()) {
			input = *matrix * input + *bias;
			elu_vec(&mut input);
		}
		input.dot(self.last) + self.last_val
		/*

		РЕЗУЛЬТАТ 0.59:
		[0.         1.15191731 2.0943951  1.67551608] [[4.1876893]] 4.765951656766298
		[0.         1.36135682 2.93215314 0.73303829] [[3.7960794]] 3.1872096715262117
		[0.20943951 0.20943951 3.35103216 1.57079633] [[0.4790966]] 0.20942622165377323
		[array([[ 1.0165445 , -0.3549156 ,  0.06285402, -0.67766607],
			   [ 0.14903155, -1.8064088 , -0.38887787,  0.16576281],
			   [-1.6730295 , -0.01575787, -0.6652524 ,  0.14698108],
			   [-0.11340991, -0.5738989 , -0.17948157,  0.07672196]],
			  dtype=float32), array([1.0014485, 4.026909 , 2.5342689, 1.7052656], dtype=float32), array([[-0.61962205,  0.35224357, -0.7050133 , -1.007992  ],
			   [-1.9364115 ,  0.20592041,  1.4574093 , -0.35939673],
			   [ 1.4148147 ,  0.6364147 , -3.2876616 ,  2.146241  ],
			   [-2.1854484 ,  1.271743  , -1.0198306 ,  0.08767886]],
			  dtype=float32), array([ 0.6362909 ,  0.22164212, -0.09167471,  0.51322347], dtype=float32), array([[-0.56710607,  3.0085683 , -1.7664146 , -0.84609425],
			   [ 1.0190496 , -0.20223191,  0.7935057 , -0.19631511],
			   [-0.4405343 ,  3.1800041 , -2.201156  , -0.6110492 ],
			   [-2.0771947 ,  1.9290844 ,  4.0726914 , -2.9167714 ]],
			  dtype=float32), array([-0.38457462, -2.7439218 , -2.7083    ,  0.58849967], dtype=float32), array([[-0.8140761 ,  0.38469988,  0.42777032, -1.3800956 ],
			   [-3.0025988 ,  0.3517017 ,  0.16559923, -2.7702587 ],
			   [ 0.85082966,  0.693335  , -0.5511722 ,  0.41399848],
			   [ 0.69392604,  0.5646706 , -2.344027  ,  0.4209389 ]],
			  dtype=float32), array([-0.66961056,  0.89743716, -1.3223268 ,  1.3417816 ], dtype=float32), array([[-1.8168594 ],
			   [ 0.81896317],
			   [-0.6204357 ],
			   [ 1.971852  ]], dtype=float32), array([0.8696187], dtype=float32)]

		*/
	}
}

fn main() {
	let keras = FromKeras::new();
	// dbg!(keras.calc(Vec4::new(0. ,         1.36135682,  2.93215314,  0.73303829)), 3.7960794);

	/*
			keras_result = 20.68706754348861
		0 has distance: 9.688962962962963
			texture_result = 0.8024540650537495
		1 has distance: 0.5128231869579011
			keras_newton_result = 0.3799791846925485
		2 has distance: 0.12376182275905621
		3 has distance: 0.06790278631445115
			texture_newton_result = 0.061825467237504525
		4 has distance: 0.034173057005849584
		5 has distance: 0.02344609293994259
		6 has distance: 0.020844051062744628
		7 has distance: 0.016087494693968203
		8 has distance: 0.01355482148546148
		9 has distance: 0.013668271569591094
		10 has distance: 0.011684613932700288
		11 has distance: 0.01064322127159081
		12 has distance: 0.009729931820595627
		13 has distance: 0.00992438289643975
		14 has distance: 0.009007786977868922
		15 has distance: 0.008943031615390723
		16 has distance: 0.008194850251712413

			keras_result = 20.68706754348861
		0 has distance: 9.9 by complete random
			texture_result = 0.8024540650537495
		1 has distance: 0.47082389671790575 by complete random
			own_texture = 0.45650341928602883
			keras_newton_result = 0.3799791846925485
			keras_newton_result = 0.31451413776505754 // for 0.59 metric
			texture_newton_result = 0.20883011810510266 // 20
		2 has distance: 0.12369597371712716 by complete random
			texture_newton_result = 0.061825467237504525
		3 has distance: 0.04960973364533557 by complete random
		4 has distance: 0.038533154441802824 by complete random
		5 has distance: 0.01990054072826446 by complete random
		6 has distance: 0.01808419625290002 by complete random
		7 has distance: 0.011215103091885559 by complete random
		8 has distance: 0.010041606583148975 by complete random
		9 has distance: 0.009628706379244263 by complete random
		10 has distance: 0.007126896719237169 by complete random
		11 has distance: 0.00693051353261683 by complete random
		12 has distance: 0.008413364278783177 by complete random
		13 has distance: 0.006797163376904864 by complete random
		14 has distance: 0.00518300542201261 by complete random
		15 has distance: 0.004835631262198832 by complete random
		16 has distance: 0.005256888782496603 by complete random
		17 has distance: 0.009610946195561293 by complete random
		18 has distance: 0.006980939041330259 by complete random
		19 has distance: 0.004608329221450163 by complete random
		20 has distance: 0.005926997020388406 by complete random
		21 has distance: 0.004331181953712468 by complete random
		22 has distance: 0.004654125731845665 by complete random
		23 has distance: 0.005875265129176049 by complete random
		24 has distance: 0.005746582520549191 by complete random
		25 has distance: 0.0037966800912218332 by complete random
		26 has distance: 0.005464902775857687 by complete random
		27 has distance: 0.0060946990023687635 by complete random
		28 has distance: 0.004571486599099566 by complete random
		29 has distance: 0.0024747751960027754 by complete random
		30 has distance: 0.004533081574093073 by complete random
		31 has distance: 0.006130299852818695 by complete random
		32 has distance: 0.005185479641283035 by complete random
		33 has distance: 0.005490820959951525 by complete random
		34 has distance: 0.004907516935255356 by complete random
		35 has distance: 0.0033522528099303563 by complete random
		36 has distance: 0.0038580096733826907 by complete random
		37 has distance: 0.004820243421589532 by complete random
		38 has distance: 0.003711505371328663 by complete random
		39 has distance: 0.005108402579866655 by complete random
		40 has distance: 0.004086233568064471 by complete random
		41 has distance: 0.0035215493436936227 by complete random
		42 has distance: 0.003907165969422204 by complete random
		43 has distance: 0.0034969629340597026 by complete random
		44 has distance: 0.0038889391910987754 by complete random
		45 has distance: 0.005280728620365306 by complete random
		46 has distance: 0.004325456797172939 by complete random
		47 has distance: 0.002708948723754203 by complete random
		48 has distance: 0.002901810009719356 by complete random
		49 has distance: 0.0027923241143069408 by complete random
		50 has distance: 0.002098249794104974 by complete random
		51 has distance: 0.004336180871022185 by complete random
		52 has distance: 0.002475622128306505 by complete random
		53 has distance: 0.0056208131120086555 by complete random
		54 has distance: 0.0024836100887274263 by complete random
		55 has distance: 0.004178861780535923 by complete random
		56 has distance: 0.0027278557761199543 by complete random
		57 has distance: 0.003145413602237131 by complete random
		58 has distance: 0.004163273838683248 by complete random
		59 has distance: 0.004251669931141886 by complete random
		60 has distance: 0.0035829251714741615 by complete random
		61 has distance: 0.0035109049199763503 by complete random
		62 has distance: 0.003038564953150786 by complete random
		63 has distance: 0.006067613527199334 by complete random
		64 has distance: 0.005327971559773478 by complete random
		65 has distance: 0.005107116786252683 by complete random
		66 has distance: 0.004896288905428629 by complete random
		67 has distance: 0.004954861070575492 by complete random
		68 has distance: 0.0033743626067290386 by complete random
		69 has distance: 0.007199910619854623 by complete random
		70 has distance: 0.007401367505374123 by complete random
		71 has distance: 0.004475903240516254 by complete random
		72 has distance: 0.0036127539566241317 by complete random
		73 has distance: 0.004739426250028965 by complete random
		74 has distance: 0.004211064564843229 by complete random
		75 has distance: 0.0026856801541609928 by complete random
		76 has distance: 0.004889664282294779 by complete random
		77 has distance: 0.005341696476252274 by complete random
		78 has distance: 0.0043257122228127914 by complete random
		79 has distance: 0.0025525886878288153 by complete random
		80 has distance: 0.004064934187914028 by complete random
		81 has distance: 0.0038824490252607467 by complete random
		82 has distance: 0.003597221957096467 by complete random
		83 has distance: 0.002568012103630183 by complete random
		84 has distance: 0.0025985267712324454 by complete random
		85 has distance: 0.003340719293223844 by complete random
		86 has distance: 0.006302539261918844 by complete random
		87 has distance: 0.0036331696395611123 by complete random
		88 has distance: 0.0038836160604273084 by complete random
		89 has distance: 0.005631917607501343 by complete random
		90 has distance: 0.00324653826694135 by complete random
		91 has distance: 0.005615701163390148 by complete random
		92 has distance: 0.004730155055728818 by complete random
		93 has distance: 0.003808705042447709 by complete random
		94 has distance: 0.003524857076922055 by complete random
		95 has distance: 0.0031784998550762494 by complete random
		96 has distance: 0.0021657296390215483 by complete random
		97 has distance: 0.0018731839148225546 by complete random
		98 has distance: 0.004051589668999117 by complete random
		99 has distance: 0.003985077708097299 by complete random
	*/

	// let points = MobiusPoints::load("points.bin");
	let sphere = MobiusPoints::sphere();

	let mut rng = rand::thread_rng();

	/*
	dbg!(points.count);

	let keras_result = MobiusPoints::distance_by_complete_random(&mut rng, |line| {
		let a = keras.calc(Vec4::new(line.o.alpha.0, line.o.beta.0, line.od.alpha.0, line.od.beta.0));
		Some(Radians(a % (2. * PI))).filter(|_| a > 0.)
	});
	dbg!(keras_result);

	let keras_newton_result = MobiusPoints::distance_by_complete_random(&mut rng, |line| {
		let a = keras.calc(Vec4::new(line.o.alpha.0, line.o.beta.0, line.od.alpha.0, line.od.beta.0));
		if a < 0. {
			None
		} else {
			let sphere = MobiusPoints::sphere();
			let ray = sphere.from_line(line);
			let mut best: Option<Approx> = None;
			if let Some(approx) = calc_mobius_from(&ray, Radians(a % (2. * PI))) {
				if approx.best_approach.t_mobius.0.abs() < 1. && approx.best_approach.distance < 5e-4 {
					best = update_best_approx(Some(approx), best);
				}
			}
			best.map(|approx| approx.best_u)
		}
	});
	dbg!(keras_newton_result);

	let texture_result = MobiusPoints::distance_by_complete_random(&mut rng, |line| {
		points.solved_points[points.get_position(&line).2].1
	});
	dbg!(texture_result);
	*/

	/*
		20:
			texture_newton_result = 0.13674342977209186

		5:
			own_texture = 0.24090247797639103
			texture_newton_result = 0.3399852240574926

		6 new:
			texture_newton_result = 0.24977750505898882
			own_texture = 0.08083557380301075

		5 new:
			own_texture = 0.07207678607873245
			texture_newton_result = 0.27890307059803077

		10:
			own_texture = 0.1314816377183247
			texture_newton_result = 0.2206097506611755

		20:
			own_texture = 0.061264383652552867
			texture_newton_result = 0.14353095125093251

		21 new:
			own_texture = 0.03800020259088548
			texture_newton_result = 0.11750452180149473

		прогноз: 90, 0.0145 (y = 1.3 / x)
		прогноз new: 90, 0.004 (y = 0.36 / x)

	*/

	let brute_force_count = 8;

	let mut new_texture =
		vec![vec![vec![vec![vec![]; brute_force_count]; brute_force_count]; brute_force_count]; brute_force_count];

	let random_count = brute_force_count * brute_force_count * brute_force_count * brute_force_count * 10;

	let mut brute_force_points = MobiusPoints::calc(brute_force_count);
	brute_force_points.save_to_texture(&format!("points{}.png", brute_force_count));

	let f = |brute_force_points: &MobiusPoints, line: &SphereLine3| {
		let (alpha1i, beta1i, alpha2i, beta2i) = brute_force_points.get_integer_angles(&line);
		let mut to_visit = Vec::with_capacity(2);
		if let Some(u) = brute_force_points.solved_points[brute_force_points
			.get_position_by_angles((alpha1i, beta1i, alpha2i, beta2i))
			.2]
			.1
		{
			to_visit.push(u);
		}
		if let Some(u) = brute_force_points.solved_points[brute_force_points
			.get_position_by_angles((alpha2i, beta2i, alpha1i, beta1i))
			.2]
			.1
		{
			to_visit.push(u);
		}
		if to_visit.is_empty() {
			to_visit.push(Radians(0.));
		}

		let mut best: Option<Approx> = None;
		for current in to_visit {
			let sphere = MobiusPoints::sphere();
			let ray = sphere.from_line(line);
			if let Some(approx) = calc_mobius_from(&ray, current) {
				if approx.best_approach.t_mobius.0.abs() < 1. && approx.best_approach.distance < 5e-4 {
					best = update_best_approx(Some(approx), best);
				}
			}
		}
		best.map(|approx| approx.best_u)
	};

	let texture_newton_result =
		MobiusPoints::distance_by_complete_random(&mut rng, |line| f(&brute_force_points, line));
	dbg!(texture_newton_result);

	let bar = ProgressBar::new(random_count as u64).with_style(
		ProgressStyle::default_bar()
			.template("[elapsed: {elapsed_precise:>8} | remaining: {eta_precise:>8} | {percent:>3}%] {wide_bar}"),
	);
	for _ in 0..random_count {
		bar.inc(1);
		let sphere_line = random_sphere_line(&mut rng);

		let (alpha1i, beta1i, alpha2i, beta2i) = brute_force_points.get_integer_angles(&sphere_line);
		if let Some(approx) = MobiusPoints::f(&sphere, &sphere_line, 100) {
			new_texture[alpha1i][beta1i][alpha2i][beta2i].push((sphere_line, approx));
		}
	}
	bar.finish();

	let bar = ProgressBar::new((brute_force_count * brute_force_count) as u64).with_style(
		ProgressStyle::default_bar()
			.template("[elapsed: {elapsed_precise:>8} | remaining: {eta_precise:>8} | {percent:>3}%] {wide_bar}"),
	);
	for (alpha1i, next_arr) in new_texture.iter().enumerate() {
		for (beta1i, next_arr) in next_arr.iter().enumerate() {
			bar.inc(1);
			for (alpha2i, next_arr) in next_arr.iter().enumerate() {
				bar.tick();
				for (beta2i, arr1) in next_arr.iter().enumerate() {
					let arr2 = &new_texture[alpha2i][beta2i][alpha1i][beta1i];

					let mut best = None;
					for (
						_,
						Approx { best_u: probably_best1, best_approach: ApproachMobiusResult { t_line: t_line1, .. } },
					) in arr1.iter()
					{
						for (
							_,
							Approx {
								best_u: probably_best2,
								best_approach: ApproachMobiusResult { t_line: t_line2, .. },
							},
						) in arr2.iter()
						{
							let mut current_sum = 0.;

							for (sphere_line, Approx { best_u: current_best, .. }) in arr1.iter().cloned().chain(
								arr2.iter()
									.cloned()
									.map(|(SphereLine3 { o, od }, approx)| (SphereLine3 { o: od, od: o }, approx)),
							) {
								let ray = sphere.from_line(&sphere_line);

								let current1 = calc_mobius_from(&ray, *probably_best1)
									.map(|Approx { best_u, .. }| (current_best.0 - best_u.0).abs())
									.unwrap_or(current_best.0);
								let current2 = calc_mobius_from(&ray, *probably_best2)
									.map(|Approx { best_u, .. }| (current_best.0 - best_u.0).abs())
									.unwrap_or(current_best.0);

								current_sum += current1.min(current2);
							}
							best = best
								.map(|(best_sum, x)| {
									if current_sum < best_sum {
										(current_sum, ((probably_best1, t_line1), (probably_best2, t_line2)))
									} else {
										(best_sum, x)
									}
								})
								.or(Some((current_sum, ((probably_best1, t_line1), (probably_best2, t_line2)))));
						}
					}

					let (_, _, pos1) = brute_force_points.get_position_by_angles((alpha1i, beta1i, alpha2i, beta2i));
					let (_, _, pos2) = brute_force_points.get_position_by_angles((alpha2i, beta2i, alpha1i, beta1i));

					if let Some((_, (a, b))) = best {
						if a.1.0 < b.1.0 {
							brute_force_points.solved_points[pos1].1 = Some(*a.0);
							brute_force_points.solved_points[pos2].1 = Some(*b.0);
						} else {
							brute_force_points.solved_points[pos1].1 = Some(*b.0);
							brute_force_points.solved_points[pos2].1 = Some(*a.0);
						}
					}
				}
			}
		}
	}
	bar.finish();

	brute_force_points.save(&format!("brute_force{}.bin", brute_force_count));
	brute_force_points.save_to_texture(&format!("brute_force{}.png", brute_force_count));

	let own_texture = MobiusPoints::distance_by_complete_random(&mut rng, |line| f(&brute_force_points, line));
	dbg!(own_texture);

	/*

	return;

	/*
	for i in 0..100 {
		let f = |line: &SphereLine3| MobiusPoints::f(&sphere, line, i);

		let mut result = 0.;
		let mut t: Duration;

		// t = time(|| result = points.distance(f));
		// println!("{} has distance: {} by brute-force with time {:?}, i, result, t);

		// t = time(|| result = points.distance_by_random(&mut rng, f));
		// println!("{} has distance: {} by random with time {:?}", i, result, t);

		t = time(|| result = MobiusPoints::distance_by_complete_random(&mut rng, f));
		println!("{} has distance: {} by complete random with time {:?}", i, result, t);
		// println!();
	}
	*/

	let matrices = 4;
	// let mut de = self_adaptive_de(vec![(-1.0, 1.0); matrices * 20], |pos| {
	// 	cost(&points, &pos.iter().map(|x| *x as f64).collect::<Vec<_>>()) as f32
	// });

	// for _ in 0..100 {
	// 	de.iter().nth(1);
	// 	let (cost, _) = de.best().unwrap();
	// 	println!("cost: {}", cost);
	// }
	// let (_, pos) = de.best().unwrap();
	// println!("pos: {:?}", pos);

	// let init = &pos.iter().map(|x| *x as f64).collect::<Vec<_>>();

	let init = [
		1.7631135,
		-0.48514795,
		0.1625297,
		-0.47643745,
		0.56052864,
		-0.75668555,
		-0.40442896,
		-1.9857948,
		0.24122548,
		-1.3028315,
		-0.32980037,
		0.29925892,
		-0.7876258,
		1.7052045,
		0.67685986,
		-0.8529413,
		-0.75659704,
		-0.18389034,
		0.41154814,
		-0.21918939,
		0.428352,
		1.8101766,
		0.90563124,
		0.10165274,
		0.92122537,
		1.2997265,
		-0.38271594,
		-0.03676629,
		0.019592285,
		-0.07131435,
		0.07439828,
		-1.016202,
		-1.5365245,
		0.015755355,
		0.008364916,
		-0.61221755,
		0.22203922,
		0.66362786,
		0.24142113,
		0.5055139,
		0.50996757,
		-0.32103267,
		-1.5047038,
		0.44143873,
		-1.8197933,
		1.1040202,
		0.7439704,
		0.96619844,
		-0.81539965,
		0.15473175,
		-0.09693393,
		0.08119172,
		0.7594962,
		-0.5477702,
		0.008321762,
		0.33817106,
		0.7284856,
		-1.7415123,
		0.45527452,
		-0.5406076,
		0.7854686,
		-1.0737889,
		-0.7066234,
		-0.07927275,
		-0.04587221,
		0.7973522,
		0.845622,
		0.2856691,
		0.22031474,
		-0.24242926,
		-0.5420022,
		1.1507479,
		0.2452383,
		-0.86461985,
		0.26165295,
		-0.226753,
		0.16423398,
		0.80947495,
		0.8001113,
		0.5667609,
	];

	if let Err(ref e) = run(matrices, points.clone(), &init) {
		println!("{}", e);
		std::process::exit(1);
	}
	*/
}


/*

Feb 25 19:09:33.202 INFO L-BFGS, max_iters: 10000
Feb 25 20:20:43.513 INFO iter: 0, cost: 0.3784942502589354, best_cost: 0.3784942502589354, cost_func_count: 1, grad_func_count: 2, jacobian_func_count: 0, hessian_func_count: 0, modify_func_count: 0, gamma: 1.0, time: 4270.310277948
Feb 25 20:52:32.432 INFO iter: 1, cost: 0.3692261770392053, best_cost: 0.3692261770392053, cost_func_count: 1, grad_func_count: 3, jacobian_func_count: 0, hessian_func_count: 0, modify_func_count: 0, gamma: 0.00000000900104335600589, time: 1908.919276647
Feb 25 22:21:42.714 INFO iter: 2, cost: 0.3627234827791401, best_cost: 0.3627234827791401, cost_func_count: 1, grad_func_count: 4, jacobian_func_count: 0, hessian_func_count: 0, modify_func_count: 0, gamma: 0.0000000000006845892819012909, time: 5350.282206213
Condition violated: "MoreThuenteLineSearch: Search direction must be a descent direction."


 */
