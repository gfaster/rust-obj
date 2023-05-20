use std::ops::{Index, IndexMut};

use super::Vertex;

#[derive(Debug)]
pub struct Tri {
    v: [Vertex; 3]
}

impl Index<usize> for Tri {
    type Output = Vertex;

    fn index(&self, index: usize) -> &Self::Output {
        self.v.index(index)
    }
}

impl IndexMut<usize> for Tri {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.v.index_mut(index)
    }
}

impl From<[Vertex; 3]> for Tri {
    fn from(value: [Vertex; 3]) -> Self {
        Self { v: value.into() }
    }
}

impl From<Tri> for [Vertex; 3] {
    fn from(value: Tri) -> Self {
        value.v
    }
}
