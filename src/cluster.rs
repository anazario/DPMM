use std::fmt::{Display, Formatter, Result};
use std::ops::{AddAssign, Index, IndexMut, SubAssign};
use ndarray::{Array1, Array2, ArrayBase};
use ndarray_inverse::Inverse;
use crate::datum::{DataSet, Datum};
use crate::traits::*;

#[derive(Default, Debug, Clone)]
pub(crate) struct Cluster<T> {
    data_index: Vec<usize>,
    data_sum: Datum<T>,
    covariance: Array2<f64>,
}

impl<T> Cluster<T>
    where
        T: PartialOrd + Copy + ToF64 + AddAssign + SubAssign
{
    pub fn new(index: usize, datum: &Datum<T>) -> Self {
        Self{
            data_index: vec![index],
            data_sum: datum.clone(),
            covariance: Array2::zeros((datum.len(), datum.len())),
        }
    }

    pub fn data_index(&self) -> &[usize] {
        self.data_index.as_slice()
    }

    pub fn covariance(&self) -> &Array2<f64>{
        &self.covariance
    }

    pub fn precision(&self) -> Option<Array2<f64>>{
        self.covariance.inv()
    }

    pub fn find(&self, index: &usize) -> bool {
        self.data_index.contains(index)
    }

    pub fn mean(&self) -> Array1<f64> {
        let data_average: Array1<f64> = ArrayBase::from_shape_fn(self.data_sum.len(), |index| {
            self.data_sum.coordinates()[index].to_f64() / self.data_index.len() as f64
        });

        data_average
    }

    pub fn set_covariance(&mut self, covariance: Array2<f64>){
        self.covariance = covariance;
    }

    pub fn add(&mut self, index: usize, datum: &Datum<T>) {
        //ensure index is a unique identifier
        if !self.data_index.contains(&index) {
            self.data_index.push(index);

            self.data_sum += datum;
        }
    }

    pub fn remove(&mut self, index: usize, datum: &Datum<T>) {
        if let Some(index) = self.data_index.iter().position(|&x| x == index) {
            self.data_index.remove(index);
            self.data_sum -= datum;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data_index.is_empty()
    }

    pub fn len(&self) -> usize{
        self.data_index.len()
    }
}

impl<T: Display> DisplayVec<usize> for Cluster<T> {
    fn get_vec(&self) -> &Vec<usize> {
        &self.data_index
    }
}

impl<T: Display> Display for Cluster<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.display_vec())
    }
}

#[derive(Default, Debug)]
pub(crate) struct ClusterList<T> {
    clusters: Vec<Cluster<T>>
}

impl<T> ClusterList<T>
    where
        T: PartialOrd + Copy + ToF64 + AddAssign + SubAssign
{
    pub fn new() -> Self{
        Self{
            clusters: Vec::new()
        }
    }

    pub fn init_all(dataset: &DataSet<T>) -> Self{

        let mut clusters: Vec<Cluster<T>> = Vec::new();

        for (index,datum) in dataset.iter(){
            let cluster = Cluster::new(*index, datum);
            clusters.push(cluster);
        }

        ClusterList{clusters}
    }

    pub fn clusters(&self) -> &[Cluster<T>] {
        self.clusters.as_slice()
    }

    pub fn len(&self) -> usize{
        self.clusters.len()
    }

    pub fn observations(&self) -> usize {
        let observations: usize = self.clusters
            .iter()
            .map(|cluster| cluster.len())
            .sum();
        observations + 1
    }

    pub fn is_empty(&self) -> bool{
        self.clusters.is_empty()
    }

    pub fn create(&mut self, index: usize, datum: &Datum<T>) {
        self.clusters.push(Cluster::new(index, datum));
    }

    pub fn remove(&mut self, index: usize, datum: &Datum<T>) -> Option<usize>{

        for (cluster_id, cluster) in self.clusters.iter_mut().enumerate() {
            if cluster.find(&index) {
                cluster.remove(index, datum);

                if cluster.is_empty() {
                    self.purge();
                    return None;
                }

                return Some(cluster_id);
            }
        }

        None
    }

    pub fn purge(&mut self) {
        self.clusters.retain(|cluster| !cluster.data_index.is_empty());
    }

    pub fn iter(&self) -> impl Iterator<Item = &Cluster<T>> {
        self.clusters.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Cluster<T>> {
        self.clusters.iter_mut()
    }
}

impl<T> FromIterator<Cluster<T>> for ClusterList<T> {
    fn from_iter<I: IntoIterator<Item = Cluster<T>>>(iter: I) -> Self {
        let clusters: Vec<Cluster<T>> = iter.into_iter().collect();
        Self { clusters }
    }
}

impl<T> Index<usize> for ClusterList<T> {
    type Output = Cluster<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.clusters[index]
    }
}

impl<T> IndexMut<usize> for ClusterList<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.clusters[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //#[test]
    fn test_cluster() {

        let d1: Datum<f32> = Datum::new(&[1.,2.,3.]);
        let d2: Datum<f32> = Datum::new(&[2.,3.,4.]);

        let mut cluster: Cluster<f32> = Cluster::new(1, &d1);
        cluster.add(45, &d2);
        println!("cluster indexes: {:?}", cluster);
        cluster.remove(45, &d2);
        println!("cluster indexes: {:?}", cluster);

    }

    #[test]
    fn test_average() {
        let mut cluster: Cluster<f32> = Cluster::new(1, &Datum::new(&[2.,3.,4.]));
        println!("{:?}", cluster.mean());
        cluster.add(45, &Datum::new(&[1.,2.,3.]));
        println!("{:?}", cluster.mean());
        println!("cluster indexes: {}", cluster);

    }

    //#[test]
    fn test_cluster_self_update(){

        let data1 = Datum::new(&[1., 2., 3.]);
        let data2 = Datum::new(&[4., 5., 6.]);
        let data3 = Datum::new(&[7., 8., 9.]);

        let data_vec = vec![&data1, &data2];

        let mut clusters = ClusterList::new();

        clusters.create(0, &data1);
        clusters.create(1, &data2);

        println!("{:#?}", clusters);

        for (index,cluster) in clusters.iter_mut().enumerate() {
            if index == 0{
                cluster.add(2, &data3);
                println!("Average: {:#?}", cluster.mean())
            }
            println!("{:#?}", cluster);
            cluster.remove(index, &data_vec[index]);
            println!("{:#?}", cluster);
        }
        clusters.purge();
        println!("{:#?}", clusters);
    }
}