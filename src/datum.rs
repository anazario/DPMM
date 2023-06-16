use std::collections::hash_map::{Values, ValuesMut};
use std::collections::HashMap;
use ndarray::{Array2, ArrayBase, Axis,};
use std::fmt::{Display, Formatter, Result};
use std::hash::Hash;
use std::ops::{AddAssign, Div, Mul, Sub};
use ndarray_stats::CorrelationExt;
use num_traits::{Float, FromPrimitive};

use crate::traits::*;

#[derive(Default, Debug, Clone)]
pub(crate) struct Datum<T> {
    coordinates: Vec<T>,
}

impl<T> DataType for Datum<T> {}

impl<T: PartialOrd+Copy> Datum<T>{
    pub fn new(point: &[T]) -> Self{
        Self{
            coordinates: point.to_vec(),
        }
    }

    pub fn coordinates(&self) -> &[T] {
        &self.coordinates
    }

    pub fn len(&self) -> usize{
        self.coordinates.len()
    }

}

impl<T: Into<f64> + Copy> Datum<T> {
    pub fn convert_to_f64(&self) -> Vec<f64> {
        self.coordinates.iter().map(|&x| (x).into()).collect()
    }
}

impl Datum<usize>{
    pub fn usize_to_f64(&self) -> Vec<f64> {
        self.coordinates.iter().map(|&x| (x) as f64).collect()
    }
}

impl<T: Mul<Output = T> + Copy> Datum<T> {
    pub fn multiply_by_index(&self, index: usize) -> Vec<T> {
        let factor = self.coordinates[index];
        self.coordinates.iter().map(|&x| x * factor).collect::<Vec<T>>()
    }
}

impl<T: Display> DisplayVec<T> for Datum<T>{
    fn get_vec(&self) -> &Vec<T> {
        &self.coordinates
    }
}

impl<T: Display> Display for Datum<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.display_vec())
    }
}

impl<T> AddAssign<&Datum<T>> for Datum<T>
    where
        T: AddAssign + Clone,
{
    fn add_assign(&mut self, other: &Datum<T>) {
        for (idx, value) in self.coordinates.iter_mut().enumerate() {
            *value += other.coordinates[idx].clone();
        }
    }
}

impl<T> Div<T> for Datum<T>
    where
        T: Into<f64> + Div<Output = T> + Copy,
{
    type Output = Datum<f64>;

    fn div(self, other: T) -> Datum<f64> {
        let coordinates: Vec<f64> = self
            .coordinates
            .into_iter()
            .map(|val| val.into() / other.into())
            .collect();

        Datum { coordinates }
    }
}

impl<T: Sub<Output = T>> Sub for Datum<T> {
    type Output = Datum<T>;

    fn sub(self, other: Datum<T>) -> Datum<T> {
        let coordinates = self
            .coordinates
            .into_iter()
            .zip(other.coordinates.into_iter())
            .map(|(x, y)| x - y)
            .collect();

        Datum {
            coordinates,
        }
    }
}

impl<T: Mul<Output = T>> Mul for Datum<T> {
    type Output = Datum<T>;

    fn mul(self, other: Datum<T>) -> Datum<T> {
        let coordinates = self
            .coordinates
            .into_iter()
            .zip(other.coordinates.into_iter())
            .map(|(x, y)| x * y)
            .collect();

        Datum {
            coordinates,
        }
    }
}

#[derive(Default, Debug)]
pub(crate) struct DataSet<T>{
    dataset: HashMap<usize, Datum<T>>,
}

impl<T: PartialOrd + Copy> DataSet<T> {
    pub fn new(data: &ArrayBase<impl ndarray::Data<Elem = T>, ndarray::Dim<[usize; 2]>>) -> Self {
        let dataset: HashMap<usize, Datum<T>> = data
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(idx, row)| (idx, Datum::new(row.as_slice().unwrap())))
            .collect();

        Self {dataset}
    }

    pub fn dataset(&self) -> &HashMap<usize, Datum<T>>{
        &self.dataset
    }

    pub fn dimension(&self) -> Option<usize>{

        if self.dataset.is_empty() {
            return None
        }

        Some(self.dataset.get(&0).unwrap().len())
    }

    pub fn is_empty(&self) -> bool{
        self.dataset.is_empty()
    }

    pub fn get(&self, key: usize) -> Option<&Datum<T>>{
        self.dataset.get(&key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&usize, &Datum<T>)> {
        self.dataset.iter()
    }

}

/*impl<T: /*Float + */AddAssign + FromPrimitive + Into<f64>> DataSet<T>{
    pub fn slice_mean(&self, slice_indexes: &[usize]) -> Datum<T>{
        let mut datum: Datum<T> = Datum { coordinates: vec![0 as f64/*T::from(0).unwrap()*/; self.dimension().unwrap()]};
        for key in slice_indexes{
            datum += self.dataset.get(key).unwrap();
        }
        datum.into()/(slice_indexes.len()).into()
    }

    pub fn slice_to_vec(&self, slice_indexes: &[usize]) -> Vec<Datum<T>>{
        slice_indexes
            .iter()
            .filter_map(|&index| self.dataset.get(&index).cloned())
            .collect()
    }

    pub fn slice_to_array(&self, slice_indexes: &[usize]) -> Array2<T>{
        let slice_vec = self.slice_to_vec(slice_indexes);
        let rows: usize = slice_vec.len();
        let cols: usize = slice_vec[0].coordinates().len();

        Array2::from_shape_vec((rows, cols), slice_vec
            .iter()
            .flat_map(|datum| datum.coordinates.iter())
            .cloned()
            .collect())
            .unwrap()
    }
} */


impl<T> IntoIterator for DataSet<T> {
    type Item = (usize, Datum<T>);
    type IntoIter = std::collections::hash_map::IntoIter<usize, Datum<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.dataset.into_iter()
    }
}

#[cfg(test)]
mod tests{
    use ndarray::Array;
    use ndarray_rand::{rand, RandomExt};
    use ndarray_rand::rand_distr::StandardNormal;
    use super::*;

    #[test]
    fn test_datum(){
        let datum = Datum::new(&[0,1,2,3]);
        println!("{}", datum);
    }

    #[test]
    fn test_div(){
        let datum = Datum::new(&[0,1,2,3]);
        println!("{}", datum/2);
    }

    /*#[test]
    fn test_dataset(){
        let data = Array::<f64, _>::random_using((10,4), StandardNormal, &mut rand::thread_rng());
        let dataset = DataSet::new(&data);

        println!("{:?}", &dataset.dimension());
        println!("Total mean: {}", &dataset.slice_mean(&[0,1]));

    }*/

    #[test]
    fn test_datum_sum(){
        let datum1 = Datum::new(&[0., 1., 2., 3.]);
        let datum2 = Datum::new(&[1., 1., 1., 1.]);
        let datum3 = Datum::new(&[1., 0., 0., 0.]);

        let mut datum_sum = Datum::new(&[0.,0.,0.,0.]);
        datum_sum += &datum1;
        println!("{:?}", datum_sum);
        datum_sum += &datum2;
        println!("{:?}", datum_sum);
        datum_sum += &datum3;
        println!("{:?}", datum_sum);

        let n = 2.;
        println!("{:?}", datum_sum/n);

        println!("{:?}", datum1.multiply_by_index(2));

    }

    /*#[test]
    pub fn test_slice_to_vec(){
        let data = Array::<f64, _>::random_using((10,4), StandardNormal, &mut rand::thread_rng());
        let dataset = DataSet::new(&data);

        let slice_vec = dataset.slice_to_vec(&[0,1,2]);

        for slice in slice_vec{
            println!("{:?}", slice);
        }
    }*/

    /*#[test]
    pub fn test_slice_to_array(){
        let data = Array::<f64, _>::random_using((2, 2), StandardNormal, &mut rand::thread_rng());
        let dataset = DataSet::new(&data);

        let slice: Vec<usize> = (0..2).collect();
        let slice_array = dataset.slice_to_array(slice.as_slice());

        println!("{:?}", &slice_array.t());
        println!("{:?}", slice_array.t().cov(1.));
    }*/

    /*#[test]
    pub fn transpose_test(){
        let data = Array::<f64, _>::random_using((2, 2), StandardNormal, &mut rand::thread_rng());
        let dataset = DataSet::new(&data);

        let slice: Vec<usize> = (0..2).collect();
        let slice_array = dataset.slice_to_array(slice.as_slice());

        println!("Array: {:?}", &slice_array);
        println!("Transpose: {:?}", &slice_array.t());
        println!("{:?}", &slice_array.t().cov(1.));

    }*/


}