use libm::{exp, pow};
use ndarray::Data;
use ndarray::prelude::*;
use ndarray_rand::{rand as rand, RandomExt};
use ndarray_rand::rand_distr::{Normal};
use ndarray_inverse::Inverse;
use hdf5::File;
use crate::datum::DataSet;
use crate::traits::ToF64;

pub fn gauss(mean: f64, variance: f64) -> f64{
    exp(-(pow(mean, 2.)/(2.*pow(variance, 2.))))
}

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

pub fn dpmm<T,V>(data: Array2<T>,
                 alpha: V,
                 mean_zero: V,
                 var_zero: Array2<V>,
                 var_data: Array2<V>)
    where
        T: ToF64 + Copy,
        V: ToF64 + Copy,
{
    // Ensure the matrices have the same shape
    assert_eq!(var_zero.shape()[0], data.shape()[1]);
    assert_eq!(var_zero.shape(), var_data.shape());

    // Ensure the matrices are square
    assert_eq!(var_zero.shape()[0], var_zero.shape()[1]);
    assert_eq!(var_data.shape()[0], var_data.shape()[1]);

    alpha.to_f64();
    mean_zero.to_f64();
    var_zero.mapv(|elem| elem.to_f64());
    var_data.mapv(|elem| elem.to_f64());

    let dataset: DataSet<T> = DataSet::new(&data);

    ()
}

#[cfg(test)]
mod tests {
    use crate::cluster::ClusterList;
    use super::*;

    //#[test]
    fn test_gen(){
        let _data = generate_1d_dataset(0., 16., 10);
        /*for datum in data {
            println!("{}", datum);
        }*/
    }

    #[test]
    fn test_dpmm(){

        let dim = 2;

        let alpha = 0.01;
        let mean_zero = 1.*Array::zeros(2);
        let var_zero = 9.*Array2::eye(2);
        let var_data = 1.*Array2::eye(2);

        println!("{:#?}", var_zero);

        // Open the HDF5 file
        let file = File::open("../test/trial_dataset1.h5").expect("Failed to open HDF5 file");

        // Load the dataset "mixed_data" into an ndarray
        let input_data: Array2<f64> = file
            .dataset("mixed_data")
            .expect("Failed to open dataset")
            .read_2d::<f64>()
            .expect("Failed to read dataset");

        let dataset = DataSet::new(&input_data);

        let start = 0;
        let end = 10;
        for (key, value) in dataset.iter().skip(start).take(end - start) {
            println!("Key: {}, Value: {}", key, value);
        }

        //let mut clusters = ClusterList::init_all(&dataset);
        let mut clusters = ClusterList::new();

        println!("cluster size: {:#?}", clusters.len());

        for (index, datum) in dataset.iter().skip(start).take(end - start){

            if clusters.is_empty(){
                clusters.create(*index, datum);
                continue;
            }

            clusters.remove(*index, datum);
            let observations = clusters.observations();

            for cluster in clusters.iter_mut(){

                let size = cluster.len();

                let prob_member = size.to_f64()/(observations.to_f64() - 1. + alpha);
                let prob_new = alpha/(observations.to_f64() - 1. + alpha);

                let cluster_precision = dataset.precision(cluster.data_index());
                println!("{:#?}", cluster_precision);
                println!("{:#?}", cluster);

                if prob_member > prob_new{
                    cluster.add(*index, datum)
                }

                println!("{}", format!("p(ci| c-i, α): {}, p(new c): {}, total = {}",
                                       prob_member, prob_new, prob_member+prob_new));


            }

            if clusters.iter_mut().find(|cluster| cluster.find(index)).is_none(){
                clusters.create(*index, datum);
            }

        }

        println!("cluster size after loop: {:#?}", clusters.len());
        println!("{:#?}", clusters);

    }

    //#[test]
    fn test_hdf5(){
        // Open the HDF5 file
        let file = File::open("../test/trial_dataset1.h5").expect("Failed to open HDF5 file");

        // Load the dataset "mixed_data" into an ndarray
        let input_data: Array2<f64> = file
            .dataset("mixed_data")
            .expect("Failed to open dataset")
            .read_2d::<f64>()
            .expect("Failed to read dataset");

        let dataset = DataSet::new(&input_data);

        println!("{}", dataset.len());
    }
}