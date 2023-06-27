#![allow(unused)]
use hdf5::File;
use ndarray::prelude::*;
use crate::datum::DataSet;
use crate::model::{DPMM, Model};

mod model;
mod datum;
mod cluster;
mod traits;
mod python;

fn main() -> hdf5::Result<()> {
    let alpha = 0.02;
    let mean= 1.*Array::zeros(2);
    let variance = 0.9*Array2::eye(2);
    let error = 0.8*Array2::eye(2);

    let parameters = Model::new(alpha, mean, variance, error);

    // Open the HDF5 file
    let file = File::open("../test/test_dataset2.h5").expect("Failed to open HDF5 file");

    // Load the dataset "mixed_data" into an ndarray
    let input_data: Array2<f64> = file
        .dataset("mixed_data")
        .expect("Failed to open dataset")
        .read_2d::<f64>()
        .expect("Failed to read dataset");

    let dataset = DataSet::new(&input_data);

    let mut dpmm = DPMM::new(dataset, parameters);

    dpmm.solve(15);
    dpmm.save_clusters_hdf5("clusters")?;

    for (index, cluster) in dpmm.clusters().iter().enumerate(){
        println!("{}", format!("cluster {} size: {}", index, cluster.len()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    /*use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::{rand, RandomExt};
    use crate::cluster::Cluster;
    //use crate::cluster::{Cluster, ClusterList};
    use crate::datum::{DataSet, Datum};
    use super::*;*/
}