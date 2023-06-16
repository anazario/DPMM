use libm::{expf, powf};
use nalgebra::{DMatrix, Vector2};
use plotters::prelude::*;
use ndarray::prelude::*;
//use ndarray_rand::{rand as rand, RandomExt};
//use ndarray_rand::rand::Rng;
//use ndarray_rand::rand_distr::{StandardNormal};
//use ndarray_stats::histogram::{strategies::Sqrt, GridBuilder};
//use noisy_float::types::{N64, n64};
//use poloto::build::{plot};

mod plot;
mod normal;
mod datum;
mod cluster;
mod traits;

/*fn main() {
    println!("Hello, world!");
}*/

const OUT_FILE_NAME: &'static str = "normal-dist.png";
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    let n_events = 10_000;

    let random_points: Vec<(f64, f64)> = {

        let mut distribution: Vec<(f64,f64)> = Vec::new();

        let x_distribution = normal::generate_1d_dataset(0., 0.06, n_events);
        let y_distribution = normal::generate_1d_dataset(0., 0.2, n_events);

        for index in 0..n_events{
            distribution.push((*x_distribution.get(index).unwrap(),
                                     *y_distribution.get(index).unwrap()));
        }

        distribution
    };

    /*samples
        .axis_iter(Axis(0))
        .map(|sample| {
            (*sample.to_vec().get(0).unwrap(),
             *sample.to_vec().get(1).unwrap())
        })
        .collect();*/

    let areas = root.split_by_breakpoints([944], [80]);

    let mut x_hist_ctx = ChartBuilder::on(&areas[0])
        .y_label_area_size(40)
        .build_cartesian_2d((-1.0..1.0).step(0.01).use_round().into_segmented(), 0..250)?;
    let mut y_hist_ctx = ChartBuilder::on(&areas[3])
        .x_label_area_size(40)
        .build_cartesian_2d(0..250, (-1.0..1.0).step(0.01).use_round())?;
    let mut scatter_ctx = ChartBuilder::on(&areas[2])
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-1f64..1f64, -1f64..1f64)?;
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;
    scatter_ctx.draw_series(
        random_points
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 2, GREEN.filled())),
    )?;
    let x_hist = Histogram::vertical(&x_hist_ctx)
        .style(GREEN.filled())
        .margin(0)
        .data(random_points.iter().map(|(x, _)| (*x, 1)));
    let y_hist = Histogram::horizontal(&y_hist_ctx)
        .style(GREEN.filled())
        .margin(0)
        .data(random_points.iter().map(|(_, y)| (*y, 1)));
    x_hist_ctx.draw_series(x_hist)?;
    y_hist_ctx.draw_series(y_hist)?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
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
    use crate::cluster::{Cluster, ClusterList};
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

    #[test]
    pub fn type_test(){
        let datum_usize: Datum<usize> = Datum::new(&[1, 2, 3]);
        let datum_i32: Datum<i32> = Datum::new(&[1, 2, 3]);
        let datum_f32: Datum<f32> = Datum::new(&[1., 2., 3.]);
        let datum_f64: Datum<f64> = Datum::new(&[1., 2., 3.]);

        /*let mut cluster = Cluster::new(0, datum_usize.usize_to_f64().as_slice());
        cluster.add(1, datum_i32.convert_to_f64().as_slice());
        cluster.add(2, datum_f32.convert_to_f64().as_slice());
        cluster.add(3, datum_f64.coordinates());*/

    }
}