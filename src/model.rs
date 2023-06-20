use std::ops::{AddAssign, SubAssign};
use libm::{exp, pow};
use ndarray::prelude::*;
use ndarray_rand::{rand as rand, RandomExt};
use ndarray_rand::rand_distr::{Normal};
use ndarray_inverse::Inverse;
use hdf5::File;
use num_traits::Float;
use crate::cluster::{Cluster, ClusterList};
use crate::datum::{DataSet, Datum};
use crate::traits::ToF64;

pub(crate) struct Model<T>{
    alpha: T,
    mean: Array1<T>,
    variance: Array2<T>,
    error: Array2<T>
}

impl<T: Float> Model<T>{
    pub fn new(alpha: T, mean: Array1<T>, variance: Array2<T>, error: Array2<T>) -> Self{
        Self{
            alpha,
            mean,
            variance,
            error,
        }
    }

    pub fn alpha(&self) -> T{
        self.alpha
    }

    pub fn mean(&self) -> &Array1<T>{
        &self.mean
    }

    pub fn variance(&self) -> &Array2<T>{
        &self.variance
    }

    pub fn error(&self) -> &Array2<T>{
        &self.error
    }

    pub fn prior_pd(&self, datum: Datum<impl ToF64 + Copy>) -> f64{
        let datum = datum.to_f64();
        let mean = &datum - &self.mean.mapv(|e| e.to_f64().unwrap());
        let variance = self.variance.mapv(|e| e.to_f64().unwrap());
        let error = self.error.mapv(|e| e.to_f64().unwrap());

        gauss(&mean, &(variance + error))
    }
}

pub(crate) struct DPMM<T> {
    dataset: DataSet<T>,
    parameters: Model<f64>,
    clusters: ClusterList<T>,
}

impl<T> DPMM<T>
    where
        T: PartialOrd + Copy + ToF64 + AddAssign + SubAssign
{
    pub fn new(dataset: DataSet<T>, parameters: Model<f64>) -> Self{
        Self{
            dataset,
            parameters,
            clusters: ClusterList::new(),
        }
    }

    pub fn dataset(&self) -> &DataSet<T>{
        &self.dataset
    }

    pub fn clusters(&self) -> &ClusterList<T>{
        &self.clusters
    }

    pub fn update_clusters(&mut self){

        for cluster in self.clusters.iter_mut(){
            cluster.set_covariance(self.dataset.cov(cluster.data_index()));
        }
    }

    fn posterior_pd(&self, datum: &Datum<T>, cluster: &Cluster<T>){

    }
}

fn gauss(mean: &Array1<f64>, variance: &Array2<f64>) -> f64 {

    let dim = mean.len();
    let mean_t = mean.t();
    let precision: Array2<f64> = match variance.inv(){
        Some(precision) => precision,
        None => Array2::zeros((dim, dim)),
    };

    exp(-mean_t.dot(&precision.dot(mean))/2 as f64)
}

pub fn posterior_pd(nk: f64, yk: &Array1<f64>, tk: &Array2<f64>,
                    mu0: &Array1<f64>, t0: &Array2<f64>, var_err: &Array2<f64>) -> f64{

    let dim = yk.len();

    let nk_tk = nk * tk;
    let den: Array2<f64> = match (&nk_tk + t0).inv() {
        Some(den) => den,
        None => Array2::zeros((dim, dim)),
    };
    println!("denominator: {:#?}", &den);

    let num: Array1<f64> = (yk.dot( &nk_tk)) + (mu0.dot(t0));
    println!("{:#?}", &num);

    let mean = &num.dot(&den);
    println!("{:#?}", mean);

    let variance = &den + var_err;
    let precision = variance.inv().unwrap();
    println!("var_err: {:#?}", var_err);
    println!("variance: {:#?}", &variance);

    let mean_t = mean.t();

    println!("{:#?}", mean_t.dot(&precision.dot(mean)));

    //gauss(mean_t.dot(&precision.dot(mean)), 1.)
    gauss(&mean, &variance)
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
        let var_zero: Array2<f64> = 9.*Array2::eye(2);
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
        let end = 5;
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
                let precision_zero = var_zero.inv().unwrap();

                let cluster_precision = dataset.precision(cluster.data_index());
                /*println!("{:#?}", cluster_precision);
                println!("{:#?}", cluster);*/

                let test = &mean_zero*precision_zero;
                println!("test multiplication: {:#?}", test);

                let mu_p = size.to_f64()*cluster.data_average()*cluster_precision;
                println!("{:#?}", mu_p);

                if prob_member > prob_new{
                    cluster.add(*index, datum)
                }

                /*println!("{}", format!("p(ci| c-i, α): {}, p(new c): {}, total = {}",
                                       prob_member, prob_new, prob_member+prob_new));*/


            }

            if clusters.iter_mut().find(|cluster| cluster.find(index)).is_none(){
                clusters.create(*index, datum);
            }

        }

        //println!("cluster size after loop: {:#?}", clusters.len());
        //println!("{:#?}", clusters);

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

    #[test]
    fn test_prob(){

        let nk = 2.;
        let avg: Array1<f64> = Array1::ones(2);
        println!("average: {:#?}", avg);

        let mean_zero: Array1<f64> = 1.*Array1::ones(2);//Array::zeros(2);
        println!("mu_0: {:#?}", mean_zero);

        let prec_zero: Array2<f64> = Array::eye(2);
        println!("t0: {:#?}", prec_zero);

        let prec_cluster: Array2<f64> = 3.*Array2::eye(2);
        println!("tk: {:#?}", &prec_cluster);

        let var_data = 1.*Array2::eye(2);

        let nk_tk = nk*&prec_cluster;
        println!("nk_tk: {:#?}", nk_tk);

        let nk_tk_plus_t0 = &nk_tk + &prec_zero;
        println!("nk_tk_plus_t0: {:#?}", nk_tk_plus_t0);

        let avg_nk_tk = &avg * &nk_tk;
        println!("avg_nk_tk: {:#?}", avg_nk_tk);

        let ppd = posterior_pd(nk, &avg, &prec_cluster, &mean_zero,
                               &prec_zero, &var_data);

        println!("probability: {}", ppd);

    }

    #[test]
    fn gauss_test(){
        let var = Array2::zeros((2,2));
        let mean = Array1::ones(2);

        println!("{:#?}", gauss(&mean, &var));
    }
}