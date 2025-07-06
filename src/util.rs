use crate::*;
use burn::prelude::Tensor;
use burn::tensor::ElementConversion;
use std::{collections::HashSet, hash::Hash};
use ndarray::Axis;
use ndarray::Array2;

pub fn vec_ok(x: Vec<f32>) -> bool {
    let n_bad: u32 = x 
        .into_iter()
        .map(|x: f32| (!x.is_finite()) as u32)
        .into_iter()
        .sum::<u32>();
    let ret = if n_bad == 0 {
        true
    } else {
        false 
    };
    return ret;
}

pub fn get_density_field(x: &Vec<f32>, linear_size: usize) -> (Vec<f32>, Vec<f32>) {
    let dist = get_dist_matrix(x, x, "periodic_signed", linear_size as f32);
    // println!("{:?}", dist);

    // let mut ret_rho_grad = [0.0; NP];
    let mut ret_rho_grad = vec![0.0f32; NP as usize];
    // let mut ret_rho = [0.0; NP];
    let mut ret_rho = vec![0.0f32; NP as usize];
    for (k, row) in dist.axis_iter(Axis(0)).enumerate() {
        // row.remove(k);
        let n_left: f32 = row 
            // .into_iter()
            // .enumerate()
            // .filter(|&(i, _)| i != k)
            // .map(|(_, v)| v)
            .map(|x: &f32| ((*x > EPSILON) & (*x < RADIUS)) as u32)
            .into_iter()
            .sum::<u32>() as f32;
        // println!("L: {:?}", n_left);
        let n_right: f32 = row 
            .map(|x: &f32| ((*x < -EPSILON) & (*x > -RADIUS )) as u32)
            .into_iter()
            .sum::<u32>() as f32;
        let rho_right = n_right / F;
        let rho_left = n_left / F;
        ret_rho[k] = rho_right + rho_left;
        ret_rho_grad[k] = (rho_right - rho_left) / (RADIUS);
    }
    if !vec_ok(ret_rho.to_vec().clone()) {
        panic!("Density field is nan! vels: {:?}" , ret_rho)
    }
    if !vec_ok(ret_rho_grad.to_vec().clone()) {
        panic!("Density grad. field is nan! vels: {:?}" , ret_rho_grad)
    }
    if ret_rho.clone().into_iter().map(|x| (x < 0.0) as u32).sum::<u32>() > 0 {
        panic!("Density field is contains negative density {:?}" , ret_rho)
    }
    // println!("RADIUSET rho: {:?}", ret_rho);
    // println!("RADIUSET rho grad: {:?}", ret_rho_grad);
    (ret_rho, ret_rho_grad)
}

pub fn get_velocity_field(x: &Vec<f32>, v: &Vec<f32>, linear_size: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dist = get_dist_matrix(x, x, "periodic_signed", linear_size as f32);
    // println!("{:?}", dist);

    let mut ret_v = vec![0.0f32; NP as usize];
    let mut ret_v_grad = vec![0.0f32; NP as usize];
    let mut ret_v_grad_second = vec![0.0f32; NP as usize];
    // let mut ret_v = [0.0; NP];
    // let mut ret_v_grad = [0.0; NP];
    // let mut ret_v_grad_second = [0.0; NP];
    for (k, row) in dist.axis_iter(Axis(0)).enumerate() {
        let n_left: u32 = row 
            .map(|x: &f32| ((*x > EPSILON) & (*x < RADIUS)) as u32)
            .into_iter()
            .sum::<u32>();
        // println!("k: {:?}", k);
        // println!("n_left: {:?}", n_left);
        let n_right: u32 = row 
            .map(|x: &f32| ((*x < -EPSILON) & (*x > -RADIUS )) as u32)
            .into_iter()
            .sum::<u32>();
        // println!("n_right: {:?}", n_right);

        let v_left: f32 = row.iter().zip(v.iter()).map(|(x,y)| {
            if *x < RADIUS && *x > EPSILON {
                *y
            } else {
                0.0
            }
        }).sum::<f32>();
        // println!("v_left: {:?}", v_left);
        let v_right: f32 = row.iter().zip(v.iter()).map(|(x,y)| {
            if *x > -RADIUS && *x < -EPSILON {
                *y
            } else {
                0.0
            }
        }).sum::<f32>();
        // println!("v_right: {:?}", v_right);
        let v_current: f32 = row.iter().zip(v.iter()).map(|(x,y)| {
            if *x > -EPSILON && *x < EPSILON {
                *y
            } else {
                0.0
            }
        }).sum::<f32>();
        // println!("v_current: {:?}", v_current);

        let vr_mean: f32 = if n_right > 0 {
            v_right / (n_right as f32)
        } else {
            0.0
        };
        let vl_mean: f32 = if n_left > 0 {
            v_left / (n_left as f32)
        } else {
            0.0
        };

        if n_left == 0 && n_right == 0 {
            ret_v[k] = 0.0;
        } else {
            ret_v[k] = (v_right + v_left) / ((n_left + n_right) as f32);
        }
        // println!("n_r = {}, n_l = {}, v_r = {}, v_l = {}", n_right, n_left, v_right, v_left);

        ret_v_grad[k] = (vr_mean - vl_mean) / (RADIUS);
        ret_v_grad_second[k] = (vr_mean + vl_mean - 2.0 * v_current) / f32::powi(RADIUS, 2);
    }
    if !vec_ok(ret_v.to_vec().clone()) {
        panic!("Velocity field is nan! vels: {:?}" , ret_v)
    }
    if !vec_ok(ret_v_grad.to_vec().clone()) {
        panic!("Velocity grad. field is nan! vels: {:?}" , ret_v_grad)
    }
    (ret_v, ret_v_grad, ret_v_grad_second)
}

pub fn product<'a: 'c, 'b: 'c, 'c, T>(
    xs: &'a [T],
    ys: &'b [T],
) -> impl Iterator<Item = (&'a T, &'b T)> + 'c {
    xs.iter().flat_map(move |x| std::iter::repeat(x).zip(ys))
}
pub fn peuclidean(x: &f32, y: &f32, l: f32, is_signed: bool) -> f32 {
    let s1 = (x - y).abs();
    let s2: f32 = s1.rem_euclid(l);
    let s3 = f32::min(s2, l - s2);
    let ret = s3.abs();
    if is_signed {
        let sign_ret = ((((s2 < l - s2) as u32) as f32) * 2.0 - 1.0) * (x-y).signum();
        ret * sign_ret
    } else {
        ret
    }
}
pub fn get_dist_matrix(xpos: &[f32], ypos: &[f32], metric: &str, l: f32) -> Array2<f32> {
    let distances: Vec<f32> = match metric {
        // Match a single value
        "euclidean_signed" => { 
            product(xpos, ypos)
                .map(|(a, b)| (a-b).signum() * (a - b).abs())
                .collect::<Vec<f32>>()
        },
        "periodic" => {
            product(xpos, ypos)
                .map(|(a, b)| peuclidean(a,b,l, false))
                .collect::<Vec<f32>>()
        },
        "periodic_signed" => {
            product(xpos, ypos)
                .map(|(a, b)| peuclidean(a,b,l, true))
                .collect::<Vec<f32>>()
        },
        _ => panic!("Not a valid distance metric is provided")
    };

    Array2::from_shape_vec((xpos.len(), ypos.len()), distances).unwrap()
}

pub fn bin_res(n: usize) -> f32 {
    2_u32.pow((largest_power_of_2_smaller_than(n) - 5).try_into().unwrap()) as f32
}
pub fn largest_power_of_2_smaller_than(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let mut power = 1;
    let mut exponent = 0;
    while power * 2 < n {
        power *= 2;
        exponent += 1;
    }
    exponent
}

pub fn is_ok<B: Backend, const D: usize>(input: Tensor<B, D>) -> bool {
    if input.clone().contains_nan().into_scalar().elem::<bool>() 
    || input.clone().sum().into_scalar().elem::<f32>().is_finite() == false {
        println!("input contains infinite/nan: {}", input);
        panic!("aaaaaaaaaaaaaa!")
    } else {
        return true
    }
}

pub fn is_set_equal<T>(a: &[T], b: &[T]) -> bool
where
    T: Eq + Hash,
{
    let a: HashSet<_> = a.iter().collect();
    let b: HashSet<_> = b.iter().collect();

    a == b
}

