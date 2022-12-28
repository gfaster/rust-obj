



#[derive(Debug, PartialEq, PartialOrd)]
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
        {
        self.x * self.x +
        self.y * self.y +
        self.z * self.z 
        }.sqrt()
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
}

#[derive(Debug, PartialEq, PartialOrd)]
pub struct TextureCoord {
    pub u: f32,
    pub v: f32
}


pub struct VertexBasic<'a> {
    pos: &'a Vec3,
    norm: Option<&'a Vec3>,
    tex: Option<&'a TextureCoord>
}

pub struct MeshDataBasic<'a> {
    v: Vec<Vec3>,
    vn: Vec<Vec3>,
    vt: Vec<TextureCoord>,
    f: Vec<VertexBasic<'a>>
}

impl<'a> MeshDataBasic<'a> {
    pub fn new() -> Self {
        MeshDataBasic { 
            v: vec![],
            vn: vec![],
            vt: vec![],
            f: vec![]
        }
    }

    pub fn add_vertex_pos(&mut self, pos: Vec3) -> usize {
        self.v.push(pos);
        self.v.len() - 1
    }

    pub fn add_vertex_normal(&mut self, norm: Vec3) -> usize {
        self.vn.push(norm);
        self.vn.len() - 1
    } 

    pub fn add_vertex_uv(&mut self, uv: TextureCoord) -> usize {
        self.vt.push(uv);
        self.vt.len() - 1
    } 

    pub fn add_tri_p(&'a mut self, pos_idx: [usize; 3]) {
        for i in 0..3 {
            self.f.push( 
                VertexBasic {
                    pos: &self.v[pos_idx[i]],
                    norm: None,
                    tex: None
                }
            );
        }
    }

    pub fn add_tri_pt(&'a mut self, pos_idx: [usize; 3], tex_idx: [usize; 3]) {
        for i in 0..3 {
            self.f.push( 
                VertexBasic {
                    pos: &self.v[pos_idx[i]],
                    norm: None,
                    tex: Some(&self.vt[tex_idx[i]])
                }
            );
        }
    }

    pub fn add_tri_pn(&'a mut self, pos_idx: [usize; 3], norm_idx: [usize; 3]) {
        for i in 0..3 {
            self.f.push( 
                VertexBasic {
                    pos: &self.v[pos_idx[i]],
                    norm: Some(&self.vn[norm_idx[i]]),
                    tex: None
                }
            );
        }
    }

    pub fn add_tri_ptn(&'a mut self, pos_idx: [usize; 3], norm_idx: [usize; 3], tex_idx: [usize; 3]) {
        for i in 0..3 {
            self.f.push( 
                VertexBasic {
                    pos: &self.v[pos_idx[i]],
                    norm: Some(&self.vn[norm_idx[i]]),
                    tex: Some(&self.vt[tex_idx[i]]),
                }
            );
        }
    }
}
