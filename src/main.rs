use hdf5::File;
use libm::{expf, powf};
use nalgebra::{DMatrix, Vector2};
use plotters::prelude::*;
use ndarray::prelude::*;
use crate::datum::DataSet;
use crate::model::{DPMM, Model};

mod plot;
mod model;
mod datum;
mod cluster;
mod traits;
mod python;

fn main() -> hdf5::Result<()>{
    let alpha = 1.;
    let mean= 1.*Array::zeros(2);
    let variance = 1.*Array2::eye(2);
    let error = 1.*Array2::eye(2);

    let parameters = Model::new(alpha, mean, variance, error);

    // Open the HDF5 file
    let file = File::open("../test/test_dataset.h5").expect("Failed to open HDF5 file");

    // Load the dataset "mixed_data" into an ndarray
    let input_data: Array2<f64> = file
        .dataset("mixed_data")
        .expect("Failed to open dataset")
        .read_2d::<f64>()
        .expect("Failed to read dataset");

    let dataset = DataSet::new(&input_data);

    let mut dpmm = DPMM::new(dataset, parameters);

    dpmm.solve(10);
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