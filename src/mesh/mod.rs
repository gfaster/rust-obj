



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


struct VertexIndexed {
    pos: u32,
    norm: Option<u32>,
    tex: Option<u32>
}

struct VertexDeindex<'a> {
    pos: &'a Vec3,
    norm: Option<&'a Vec3>,
    tex: Option<&'a TextureCoord>
}

pub struct Vertex {
    pos: Vec3,
    normal: Vec3,
    binormal: Vec3,
    tangent: Vec3,
    tex: TextureCoord
}

pub struct MeshDataBasic {
    v: Vec<Vec3>,
    vn: Vec<Vec3>,
    vt: Vec<TextureCoord>,
    f: Vec<VertexIndexed>
}

pub enum MeshError {
    VertexPositionIndexInvalid {tried: u32, max: u32},
    VertexNormalIndexInvalid {tried: u32, max: u32},
    VertexTextureIndexInvalid {tried: u32, max: u32},
}

impl MeshDataBasic {
    pub fn new() -> Self {
        MeshDataBasic { 
            v: vec![],
            vn: vec![],
            vt: vec![],
            f: vec![]
        }
    }

    /// adds a new vertex position, and returns the index
    /// this function is unaware of duplicates - it's up 
    /// to the caller to be efficent with memory
    pub fn add_vertex_pos(&mut self, pos: Vec3) -> usize {
        self.v.push(pos);
        self.v.len() - 1
    }

    /// adds a new vertex normal, and returns the index
    /// this function is unaware of duplicates - it's up 
    /// to the caller to be efficent with memory
    pub fn add_vertex_normal(&mut self, norm: Vec3) -> usize {
        self.vn.push(norm);
        self.vn.len() - 1
    } 

    /// adds a new vertex texture coordinate, and returns the index
    /// this function is unaware of duplicates - it's up 
    /// to the caller to be efficent with memory
    pub fn add_vertex_uv(&mut self, uv: TextureCoord) -> usize {
        self.vt.push(uv);
        self.vt.len() - 1
    } 

    /// gets references to the actual Vec3 comprising verticies
    /// returns a MeshError if any index is out of bounds
    fn deref_vertex(&self, vtx: &VertexIndexed) -> Result<VertexDeindex, MeshError> {
        Ok (VertexDeindex { 
            pos: self.v.get(vtx.pos as usize)
                .ok_or_else(|| MeshError::VertexPositionIndexInvalid { tried: vtx.pos, max: (self.v.len() - 1) as u32 })?,
            norm: match vtx.norm {
                Some(vn) => Some(self.vn.get(vn as usize)
                    .ok_or_else(|| MeshError::VertexNormalIndexInvalid { tried: vn, max: (self.vn.len() - 1) as u32})?),
                None => None
            },
            tex: match vtx.tex {
                Some(vt) => Some(self.vt.get(vt as usize)
                    .ok_or_else(|| MeshError::VertexTextureIndexInvalid { tried: vt, max: (self.vt.len() - 1) as u32})?),
                None => None
            }
        })
    }


    pub fn add_tri(&mut self, vtx: VertexIndexed) -> Result<(), MeshError> {
        let _ = self.deref_vertex(&vtx)?;

        self.f.push(vtx);

        Ok(())
    }


}
