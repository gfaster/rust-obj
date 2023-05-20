use std::ops::{Index, IndexMut};

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct TextureCoord {
    pub u: f32,
    pub v: f32
}

impl From<[f32; 2]> for TextureCoord {
    fn from(value: [f32; 2]) -> Self {
        Self { u: value[0], v: value[1] }
    }
}

impl From<TextureCoord> for [f32; 2] {
    fn from(value: TextureCoord) -> Self {
        [value.u, value.v]
    }
}

impl Index<usize> for TextureCoord {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.u,
            1 => &self.v,
            _ => panic!("index {index} is out of bounds")
        }
    }
}

impl IndexMut<usize> for TextureCoord {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.u,
            1 => &mut self.v,
            _ => panic!("index {index} is out of bounds")
        }
    }
}
