use libm::{expf, powf};
use nalgebra::{DMatrix, Vector2};
use plotters::prelude::*;
use ndarray::prelude::*;

mod plot;
mod model;
mod datum;
mod cluster;
mod traits;

fn main() {
    println!("Hello, world!");
}

fn gauss(mean: f32, variance: f32) -> f32{
    expf(-(powf(mean, 2.)/(2.*powf(variance, 2.))))
}

fn mean(data: &[f32]) -> Option<f32>{
    let total = data.len();

    if total == 0{
        return None
    }
    Some(data.iter().sum::<f32>()/total as f32)
}

fn _sq_mat_test(side: usize){
    let mat = DMatrix::<f32>::identity(side,side);
    println!("{}", mat.determinant());
}

fn mat_mult_test(){
    let mat = DMatrix::<f32>::from_row_slice(2,2, &[
        2., 0.,
        0., 2.,
    ]);

    let v= Vector2::from_row_slice(&[1.,1.]);
    let vt = v.transpose();

    println!("{}", gauss((vt*mat*v).x, 1.));
}

/*fn sampling(mean: f32, std_dev: f32, num_samples: usize) -> Result<(), Box<dyn std::error::Error>>{

    let _samples = Array::<f64, _>::random_using((10000,2), StandardNormal, &mut rand::thread_rng());
    /*let data = samples.mapv(|e| n64(e));
    let grid = GridBuilder::<Sqrt<N64>>::from_array(&data).unwrap().build();
    let histogram = data.histogram(grid);
    let histogram_matrix = histogram.counts();
    let data = histogram_matrix.sum_axis(Axis(0));
    let histogram_data: Vec<(f32, f32)> = data.iter().enumerate().map(|(e, i)| (e as f32, *i) ).collect();
*/
    let _file = std::fs::File::create("standard_normal_hist.svg").unwrap();
    let _graph = plot("Histogram");
    //graph.histogram(histogram_data);
    //graph.histogram(histogram_data).xmarker(0).ymarker(0);
    //graph.simple_theme(poloto::upgrade_write(file));

    Ok(())
}*/

fn arr_test(){
    let arr0 = array![[1., 2., 3.], [ 4., 5., 6.]];
    let arr = Array::from_elem((2, 1), 1.);
    println!("{}", arr0+arr);
}

#[cfg(test)]
mod tests{
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::{rand, RandomExt};
    use crate::cluster::Cluster;
    //use crate::cluster::{Cluster, ClusterList};
    use crate::datum::{DataSet, Datum};
    use super::*;

    #[test]
    fn test_1(){
        println!("The mean is {}", mean(&[1.,2.,3.,4.]).unwrap());
        mat_mult_test();
    }

    /*#[test]
    fn test_sampling(){
        sampling(0., 1., 1000);
    }*/

    #[test]
    fn ndarray_test(){
        arr_test();
    }

    /*#[test]
    pub fn test_data_manager(){
        let data = Array::<f64, _>::random_using((10,2), StandardNormal, &mut rand::thread_rng());
        println!("{:#?}", data);

        let dataset = DataSet::new(&data);
        println!("{:#?}", dataset);

        let indexes = [0, 1, 5, 6, 8];

        let mut clusters = ClusterList::new();

        /*for index in indexes{
            clusters.create(index, dataset.get(index).unwrap().convert_to_f64())
        }*/

        for (index,data) in dataset.iter(){
            clusters.create(*index, data.convert_to_f64().as_slice());
        }

        println!("{:#?}", clusters);

        for (index, cluster) in clusters.iter_mut().enumerate(){
            if index == 2{
                println!("{:#?}",cluster.data_average());
                cluster.add(index, dataset.get(index).unwrap().convert_to_f64().as_slice());
                println!("{:#?}",cluster.data_average());
                /*cluster.remove(index, dataset.get(index).unwrap().convert_to_f64().as_slice());
                println!("{:#?}",cluster.data_average());*/
            }
        }

        println!("{:#?}", clusters);

    }*/

    /*#[test]
    pub fn type_test(){
        let datum_usize: Datum<usize> = Datum::new(&[1, 2, 3]);
        let datum_i32: Datum<i32> = Datum::new(&[1, 2, 3]);
        let datum_f32: Datum<f32> = Datum::new(&[1., 2., 3.]);
        let datum_f64: Datum<f64> = Datum::new(&[1., 2., 3.]);

        let mut cluster = Cluster::new(0, datum_usize.usize_to_f64().as_slice());
        cluster.add(1, datum_i32.convert_to_f64().as_slice());
        cluster.add(2, datum_f32.convert_to_f64().as_slice());
        cluster.add(3, datum_f64.coordinates());

    }*/
}