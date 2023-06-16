use std::cell::RefCell;
use std::fmt::{Display, Formatter, Result};
use std::ops::{AddAssign, Div};
use crate::datum::Datum;
use crate::traits::{DataType, DisplayVec};
use num_traits::{Float, FromPrimitive,NumCast};

#[derive(Default, Debug, Clone)]
pub(crate) struct Cluster {
    data_index : Vec<usize>,
    data_sum: Vec<f64>,
}

impl Cluster {
    pub fn new(index: usize, datum: &[f64]) -> Self{
        Self{
            data_index: vec![index],
            data_sum: datum.to_vec(),
        }
    }

    pub fn data_index(&self) -> &[usize]{
        self.data_index.as_slice()
    }

    pub fn data_average(&self) -> Vec<f64>{
        let vec_avg: Vec<f64> = self.data_sum
            .iter()
            .map(|e| e/self.data_index.len() as f64)
            .collect();

        vec_avg
    }

    pub fn find(&self, index: &usize) -> bool{
        self.data_index.contains(index)
    }

    pub fn add(&mut self, index: usize, datum: &[f64]){
        //ensure index is a unique identifier
        if !self.data_index.contains(&index) {
            self.data_index.push(index);

            self.data_sum = self.data_sum
                .iter()
                .enumerate()
                .map(|(index, point)| point+datum[index])
                .collect::<Vec<f64>>();
        }
    }

    pub fn remove(&mut self, index: usize, datum: &[f64]){
        if let Some(index) = self.data_index.iter().position(|&x| x == index) {
            self.data_index.remove(index);

            self.data_sum = self.data_sum
                .iter()
                .enumerate()
                .map(|(index, point)| point-datum[index])
                .collect::<Vec<f64>>();
        }
    }
}

impl DisplayVec<usize> for Cluster{
    fn get_vec(&self) -> &Vec<usize> {
        &self.data_index
    }
}

impl Display for Cluster {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.display_vec())
    }
}

#[derive(Default, Debug)]
pub(crate) struct ClusterList{
    clusters: Vec<Cluster>
}

impl ClusterList{
    pub fn new() -> Self{
        Self{
            clusters: Vec::new()
        }
    }

    pub fn clusters(&self) -> &[Cluster]{
        self.clusters.as_slice()
    }

    pub fn create(&mut self, index: usize, datum: &[f64]){
        self.clusters.push(Cluster::new(index, datum));
    }

    pub fn remove(&mut self, index: usize, datum: &[f64]){
        for cluster in self.clusters.iter_mut(){
            if cluster.find(&index){
                cluster.remove(index, datum);
                self.purge();
                break;
            }
        }
    }

    pub fn purge(&mut self){
        self.clusters.retain(|cluster| !cluster.data_index.is_empty());
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Cluster> {
        self.clusters.iter_mut()
    }
}

impl IntoIterator for ClusterList {
    type Item = Cluster;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.clusters.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /*#[test]
    fn test_cluster() {
        let mut cluster: Cluster = Cluster::new(1, &[2.,3.,4.]);
        cluster.add(45, &[1.,2.,3.]);
        println!("cluster indexes: {:?}", cluster);
        cluster.remove(45, &[1.,2.,3.]);
        println!("cluster indexes: {:?}", cluster);

    }*/

    /*#[test]
    fn test_average(){
        let mut cluster: Cluster = Cluster::new(1, &[2.,3.,4.]);
        println!("{:?}", cluster.data_average());
        cluster.add(45, &[1.,2.,3.]);
        println!("{:?}", cluster.data_average());
        println!("cluster indexes: {}", cluster);

    }*/

    #[test]
    fn test_cluster_self_update(){

        let data1 = &[1., 2., 3.];
        let data2 = &[4., 5., 6.];
        let data3 = &[7., 8., 9.];

        let data_vec = vec![data1, data2];

        /*let mut clusters = ClusterList::new();

        clusters.create(0, data1);
        clusters.create(1, data2);

        println!("{:#?}", clusters);

        for (index,cluster) in clusters.iter_mut().enumerate(){
            if index == 0{
                cluster.add(2, data3);
            }
            println!("{:#?}", cluster);
            cluster.remove(index, data_vec[index]);
            println!("{:#?}", cluster);
        }
        clusters.purge();
        println!("{:#?}", clusters);*/
    }
}