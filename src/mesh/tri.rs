use std::ops::{Index, IndexMut};

use super::Vertex;
use crate::mesh;

#[derive(Debug)]
pub struct Tri {
    v: [Vertex; 3],
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
        Self { v: value }
    }
}

impl From<Tri> for [Vertex; 3] {
    fn from(value: Tri) -> Self {
        value.v
    }
}

impl Tri {
    pub fn calculate_normal(&self) -> mesh::Vec3 {
        let v = &self.v;
        let edge1 = v[0].pos - v[1].pos;
        let edge2 = v[1].pos - v[2].pos;
        edge1.cross(&edge2).normalize()
    }
}
