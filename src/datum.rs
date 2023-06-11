use std::fmt::{Display, Formatter, Result};
use crate::traits::DisplayVec;

#[derive(Default)]
pub struct Datum<T> {
    coordinates: Vec <T>,
    cluster_id: Option<usize>,
}

impl<T: Copy> Datum<T>{
    pub fn new(point: &[T]) -> Self{
        Self{
            coordinates: point.to_vec(),
            cluster_id: None,
        }
    }

    pub fn len(&self) -> usize{
        self.coordinates.len()
    }

    pub fn cluster_id(&self) -> Option<usize>{
        self.cluster_id
    }

    pub fn assign_to(&mut self, id: usize){
        self.cluster_id = Some(id)
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

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_datum(){
        let datum = Datum::new(&[0,1,2,3]);
        println!("{}", datum);
    }
}