use std::fmt::{Display, Formatter, Result};
use std::ptr::addr_of_mut;
use crate::datum::{Datum};
use crate::traits::DisplayVec;

#[derive(Default)]
struct Cluster {
   data_index : Vec<usize>,
}

impl Cluster{
    pub fn new(index: usize) -> Self{
        Self{
            data_index: vec![index],
        }
    }

    pub fn add(&mut self, index: usize){
        //ensure index is a unique identifier
        if !self.data_index.contains(&index) {
            self.data_index.push(index)
        }
    }

    pub fn remove(&mut self, index: usize){
        if let Some(index) = self.data_index.iter().position(|&x| x == index) {
            self.data_index.remove(index);
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster() {
        let mut cluster = Cluster::new(1);
        cluster.add(45);
        println!("cluster indexes: {}", cluster);
    }
}