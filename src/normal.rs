use ndarray::prelude::*;
//use ndarray_rand::rand::Rng;
use ndarray_rand::{rand as rand, RandomExt};
use ndarray_rand::rand_distr::{Normal};

/*trait Distribution{

}*/

/*struct Gaussian{
    mean: f64,
    std_dev: f64
}*/

pub fn generate_1d_dataset(mean: f64, std_dev: f64, events: usize) -> Vec<f64>{

    if events <= 0 {
        panic!("{}",format!("The number of events can only be a positive integer!"));
    }

    let norm_dist = Normal::new(mean, std_dev).unwrap();
    let samples = Array::<f64, _>::random_using((events,1), norm_dist, &mut rand::thread_rng());

    samples.axis_iter(Axis(0))
           .map(|sample| {
               *sample.to_vec().get(0).unwrap()
           }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen(){
        let _data = generate_1d_dataset(0., 16., 10);
        /*for datum in data {
            println!("{}", datum);
        }*/
    }
}