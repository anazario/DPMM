use std::fmt::Debug;
use std::io;
use std::io::Write;
use std::iter::Sum;
use std::ops::{AddAssign, SubAssign};
use libm::{exp, pow};
use ndarray::prelude::*;
use ndarray_rand::{rand as rand, RandomExt};
use ndarray_rand::rand_distr::{Normal};
use ndarray_inverse::Inverse;
use hdf5::{File, H5Type};
use ndarray::ScalarOperand;
use num_traits::{Float, ToPrimitive};
use crate::cluster::{Cluster, ClusterList};
use crate::datum::{DataSet, Datum};
use crate::traits::ToF64;

#[derive(Default, Debug)]
pub(crate) struct Model<T>{
    alpha: T,
    mean: Array1<T>,
    variance: Array2<T>,
    error: Array2<T>
}

impl<T> Model<T>
    where
        T: Float + Debug + ScalarOperand + Sum + AddAssign{

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

    pub fn precision(&self) -> Array2<T> {
        self.variance.inv().expect("Given variance must have an inverse!")
    }

    pub fn error(&self) -> &Array2<T>{
        &self.error
    }

    pub fn prior_pd(&self, datum: &Datum<impl ToF64 + Copy>) -> f64{
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
        T: PartialOrd + Copy + ToF64 + AddAssign + SubAssign + Debug
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

    pub fn update_clusters(&mut self) {

        for cluster in self.clusters.iter_mut(){

            let cov = self.dataset.cov(cluster.data_index());
            cluster.set_covariance(cov);
        }
    }

    fn posterior_pd(&self, datum: &Datum<T>, cluster: &Cluster<T>) -> f64{

        let dim = datum.len();

        let mk = cluster.mean();
        let nk = self.clusters.observations() as f64;
        let tk = match cluster.precision() {
            Some(precision) => precision,
            None => Array2::zeros((dim, dim)),
        };
        let num = &mk.dot(&(nk * &tk)) + self.parameters.mean().dot(&self.parameters.precision());
        let den = nk * &tk + self.parameters.precision();

        let precision = den.inv().expect("Posterior precision does not exist!");

        let posterior_mean = num.dot(&precision);
        let posterior_variance = precision + self.parameters.error();

        let mean = datum.to_f64() - posterior_mean;

        gauss(&mean, &posterior_variance)
    }

    pub fn solve(&mut self, iterations: usize){

        let alpha = self.parameters.alpha();

        /*let start = 0;
        let end = 1000;*/
        for i in 0..iterations {
            print!("processing iteration {i}");
            io::stdout().flush().unwrap();
            for (index, datum) in self.dataset.iter() {//.skip(start).take(end - start) {

                if self.clusters.is_empty(){
                    self.clusters.create(*index, datum);
                    continue;
                }

                //remove new observation from cluster and update the clusters;
                let mod_cluster_id = self.clusters.remove(*index, datum);

                //if an observation has been removed from a cluster, update that cluster
                if let Some(index) = mod_cluster_id {
                    let data_indexes = self.clusters[index].data_index();
                    let new_covariance = self.dataset.cov(data_indexes);
                    self.clusters[index].set_covariance(new_covariance);
                };

                //calculate all cluster assignment/ new cluster probabilities
                let observations = self.clusters.observations();
                let p_create = self.parameters.prior_pd(datum) * alpha/(observations as f64 - 1. + alpha);
                let p_assign = self.clusters
                    .iter()
                    .map(|cluster| {
                        let cluster_assignment = cluster.len() as f64 / (observations as f64 - 1. + alpha);
                        self.posterior_pd(datum, cluster)*cluster_assignment
                    })
                    .collect::<Vec<f64>>();

                /*println!("probabilities of cluster assignments: {:#?}", p_assign);
                println!("probability of new cluster: {:#?}", p_create);*/

                //find cluster with the highest assignment probability
                let (max_index, max_value) = match p_assign
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
                    Some((index, value)) => (index, value),
                    None => panic!("No probabilities assigned to cluster!"),
                };

                if max_value > &p_create{
                    self.clusters[max_index].add(*index, datum);
                }
                else{
                    self.clusters.create(*index, datum)
                }
            }
            print!("\r");
        }

        println!("\ntotal clusters: {:#?}", self.clusters.len())

    }

    pub fn cluster_data(&self) -> Vec<Array2<T>>{

        if self.clusters.is_empty(){
            panic!("No data in clusters!");
        }

        let mut cluster_data: Vec<Array2<T>> = Vec::new();

        for cluster in self.clusters.iter(){
            let datum_indexes = cluster.data_index();
            cluster_data.push(self.dataset.slice_to_array(datum_indexes));
        }

        cluster_data
    }
}

impl<T> DPMM<T>
    where
        T: PartialOrd + Copy + ToF64 + AddAssign + SubAssign + Debug + H5Type{

    pub fn save_clusters_hdf5(&self, filename: &str) -> hdf5::Result<()>{
        // Open the HDF5 file
        let file = File::create(filename.to_owned()+".h5")?;

        // Create a group to store the datasets
        let group = file.create_group("datasets")?;

        for (i, array) in self.cluster_data().iter().enumerate(){
            let dataset_name = format!("cluster_{}", i);

            // Create the dataset in the group
            let dataset = group
                .new_dataset::<f64>()
                .shape(array.shape())
                .create(dataset_name.as_str())
                .expect("No dataset found!");

            // Write the array data to the dataset
            dataset.write(array)?;
        }

        println!("created file {}.h5", filename);
        Ok(())
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

#[cfg(test)]
mod tests {
    use ndarray_rand::rand::Rng;
    use crate::cluster::ClusterList;
    use super::*;

    //#[test]
    fn test_gen(){
        let _data = generate_1d_dataset(0., 16., 10);
        /*for datum in data {
            println!("{}", datum);
        }*/
    }

    /*#[test]
    fn test_model(){
        let alpha = 0.01;
        let mean_zero= 1.*Array::zeros(2);
        let var_zero = 9.*Array2::eye(2);
        let var_data = 1.*Array2::eye(2);

        let parameters = Model{
            alpha,
            mean: mean_zero,
            variance: var_zero,
            error: var_data,
        };

        // Open the HDF5 file
        let file = File::open("../test/trial_dataset1.h5").expect("Failed to open HDF5 file");

        // Load the dataset "mixed_data" into an ndarray
        let input_data: Array2<f64> = file
            .dataset("mixed_data")
            .expect("Failed to open dataset")
            .read_2d::<f64>()
            .expect("Failed to read dataset");

        let dataset = DataSet::new(&input_data);

        let mut dpmm = DPMM::new(dataset, parameters);

        dpmm.solve(100);
    }*/

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