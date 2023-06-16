use std::fmt::Display;

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

pub(crate) trait DataType {}