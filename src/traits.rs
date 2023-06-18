use std::fmt::Display;
use ndarray::Array2;

pub(crate) trait DisplayVec<T: Display> {
    fn display_vec(&self) -> String {
        // Default implementation for displaying Vec
        let elements: Vec<String> = self
            .get_vec()
            .iter()
            .map(|x| x.to_string())
            .collect();
        format!("({})", elements.join(", "))
    }

    fn get_vec(&self) -> &Vec<T>;
}

// Trait for converting Datum<T> to f64
pub trait ToF64 {
    fn to_f64(&self) -> f64;
}

// Implement ToF64 for usize
impl ToF64 for usize {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

// Implement ToF64 for i32
impl ToF64 for i32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

// Implement ToF64 for f32
impl ToF64 for f32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

// Implement ToF64 for f64
impl ToF64 for f64 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

