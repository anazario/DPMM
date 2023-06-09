use libm::{expf, powf};

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

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_1(){
        println!("{}", mean(vec![1.,2.,3.].as_slice()).unwrap());
    }
}