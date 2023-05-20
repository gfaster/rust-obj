use std::ops::IndexMut;
use std::ops::Index;


#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

impl Vec3 {
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 { 
            x: (self.y * other.z) - (self.z * other.y),
            y: (self.z * other.x) - (self.x * other.z),
            z: (self.x * other.y) - (self.y * other.x)
        }
    }

    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3 { 
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z
        }
    }

    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3 { 
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z
        }
    }

    pub fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x +
        self.y * other.y +
        self.z * other.z 
    }

    pub fn sq_mag(&self) -> f32 {
        self.x * self.x +
        self.y * self.y +
        self.z * self.z 
    }

    pub fn mag(&self) -> f32 {
        self.sq_mag().sqrt()
    }

    pub fn scale(&self, scale: f32) -> Vec3 {
        Vec3 { 
            x: self.x * scale, 
            y: self.y * scale, 
            z: self.z * scale 
        }
    }

    pub fn normalized(&self) -> Result<Vec3, &str> {
        let sq_mag = self.sq_mag();
        if sq_mag == 0.0 {
            return Err("Cannot normalize zero length vector");
        };
        let mag = sq_mag.sqrt();

        Ok(Vec3 { 
            x: self.x / mag, 
            y: self.y / mag, 
            z: self.z / mag 
        })
    }


    pub fn orthoginal(&self) -> Result<Vec3, &str> {
        let sq_mag = self.sq_mag();
        if sq_mag == 0.0 {
            return Err("Cannot find a vector orthoginal to the zero vector");
        }
        let tangent = {
            // will be colinear when self = k * <1, -1, 1>
            let v1 = Vec3{x: -self.z, y: self.x, z: self.y};
            let t = self.cross(&v1);
            if t.sq_mag() < sq_mag * 0.02 {
                let v2 = Vec3{x: -self.y, y: self.z, z: self.x};
                self.cross(&v2)
            } else {
                t
            }
        };
        Ok(tangent)
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index {index} is out of bounds")
        }
    }
}

impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index {index} is out of bounds")
        }
    }
}

impl From<[f32; 4]> for Vec3 {
    fn from(value: [f32; 4]) -> Self {
        Self { x: value[0], y: value[1], z: value[2] }
    }
}

impl From<Vec3> for [f32; 3] {
    fn from(value: Vec3) -> Self {
        [
            value.x,
            value.y,
            value.z
        ]
    }
}

impl From<Vec3> for [f32; 4] {
    fn from(value: Vec3) -> Self {
        [
            value.x,
            value.y,
            value.z,
            1.0
        ]
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(value: [f32; 3]) -> Self {
        Self { x: value[0], y: value[1], z: value[2] }
    }
}
