use std::collections::HashMap;
use ndarray::{Array2, ArrayBase, Axis,};
use std::fmt::{Display, Formatter, Result};
use std::ops::{AddAssign, Div, Mul, Sub, SubAssign};
use ndarray_stats::CorrelationExt;
use ndarray_inverse::Inverse;

use crate::traits::*;

#[derive(Default, Debug, Clone)]
pub(crate) struct Datum<T> {
    coordinates: Vec<T>,
}

impl<T: Copy + ToF64> Datum<T>{
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

// Implementation for Datum<T>
impl<T: ToF64> Datum<T> {
    pub fn to_f64(&self) -> Vec<f64> {
        self.coordinates.iter().map(|value| value.to_f64()).collect()
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
        T: AddAssign + Copy,
{
    fn add_assign(&mut self, rhs: &Datum<T>) {
        // Ensure the sizes match
        assert_eq!(self.coordinates.len(), rhs.coordinates.len());

        // Perform element-wise addition
        for (idx, value) in self.coordinates.iter_mut().enumerate() {
            *value += rhs.coordinates[idx];
        }
    }
}

impl<T> SubAssign<&Datum<T>> for Datum<T>
    where
        T: SubAssign + Copy,
{
    fn sub_assign(&mut self, rhs: &Datum<T>) {
        // Ensure the sizes match
        assert_eq!(self.coordinates.len(), rhs.coordinates.len());

        // Perform element-wise subtraction
        for (idx, value) in self.coordinates.iter_mut().enumerate() {
            *value -= rhs.coordinates[idx];
        }
    }
}

impl<T> Div<T> for Datum<T>
    where
        T: ToF64 + Div<Output = T> + Copy,
{
    type Output = Datum<f64>;

    fn div(self, other: T) -> Datum<f64> {
        let coordinates: Vec<f64> = self
            .coordinates
            .into_iter()
            .map(|val| val.to_f64() / other.to_f64())
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

impl<T: Copy + ToF64> DataSet<T> {
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

    pub fn len(&self) -> usize{
        self.dataset().len()
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

    pub fn slice_to_array(&self, slice_indexes: &[usize]) -> Array2<T> {
        let rows = slice_indexes.len();
        let cols = self.dataset.get(&slice_indexes[0]).map(|datum| datum.len()).unwrap_or(0);

        let data: Vec<_> = slice_indexes
            .iter()
            .filter_map(|&index| self.dataset.get(&index).map(|datum| &datum.coordinates))
            .flat_map(|coordinates| coordinates.iter())
            .cloned()
            .collect();

        Array2::from_shape_vec((rows, cols), data).unwrap()
    }

    pub fn cov(&self, slice_indexes: &[usize]) -> Array2<f64> {
        let data_subset = self.slice_to_array(slice_indexes).mapv(|elem| elem.to_f64());

        if slice_indexes.len() == 1 {
            return data_subset.t().cov(0.).unwrap()
        }
        data_subset.t().cov(1.).unwrap()
    }

    pub fn precision(&self, slice_indexes: &[usize]) -> Array2<f64>{
        let cov = self.cov(slice_indexes);

        if let Some(result) = cov.inv(){
            return result
        }

        let dim = self.dimension().unwrap();
        Array2::zeros((dim,dim))
    }
}

#[cfg(test)]
mod tests{
    use ndarray::Array;
    use ndarray_rand::{rand, RandomExt};
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_stats::CorrelationExt;
    use super::*;

    //#[test]
    fn test_datum(){
        let datum = Datum::new(&[0,1,2,3]);
        println!("{}", datum);
    }

    //#[test]
    fn test_div(){
        let datum = Datum::new(&[0,1,2,3]);
        println!("{}", datum/2);
    }

    //#[test]
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

    //#[test]
    pub fn test_cov(){
        let data = Array::<f32, _>::random_using((4, 2), StandardNormal, &mut rand::thread_rng());
        let dataset = DataSet::new(&data);

        let slice: Vec<usize> = (0..4).collect();
        let slice_array = dataset.slice_to_array(slice.as_slice());

        println!("{:?}", &slice_array);
        println!("{:?}", &slice_array.t());
        println!("{:?}", &slice_array.t().cov(0.));
        println!("{:?}", &slice_array.t().cov(1.));

        let cov = slice_array.t().cov(1.).unwrap();

        println!("{:?}", cov.inv());
    }

    //#[test]
    fn test_data(){
        let data = Array::<f32, _>::random_using((4, 2), StandardNormal, &mut rand::thread_rng());
        let dataset = DataSet::new(&data);

        for data in dataset.iter(){
            dataset.get(*data.0);
        }
    }
}