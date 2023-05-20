#![allow(dead_code)]

use std::collections::HashMap;

mod tri;
pub use tri::*;

mod vec3;
pub use vec3::*;

mod texcoord;
pub use texcoord::*;


/// row major currently, want to switch to column major eventually
pub struct Mat3 {
    d: [f32; 9]
}

impl Mat3 {
    pub fn new_from_columns(v1: &Vec3, v2: &Vec3, v3: &Vec3) -> Mat3 {
        Mat3 { d: [
            v1.x, v2.x, v3.x,
            v1.y, v2.y, v3.y,
            v1.z, v2.z, v3.z
        ] }
    }

    pub fn basis_from_up(positive_z: &Vec3) -> Mat3 {
        let tangent = positive_z.orthoginal().unwrap();
        let bitangent = positive_z.cross(&tangent);
        Self::new_from_columns(&tangent, &bitangent, positive_z)
    }

    fn get_column(&self, idx: usize) -> Vec3 {
        Vec3 { 
            x: self.d[0 + idx],
            y: self.d[3 + idx],
            z: self.d[6 + idx]
        }
    }

    fn get_row(&self, idx: usize) -> Vec3 {
        Vec3 { 
            x: self.d[0 + idx * 3],
            y: self.d[1 + idx * 3],
            z: self.d[2 + idx * 3]
        }
    }

    pub fn apply_to_vec(&self, rhs: &Vec3) -> Vec3 {
        Vec3 { 
            x: self.get_row(0).dot(&rhs),
            y: self.get_row(1).dot(&rhs),
            z: self.get_row(2).dot(&rhs)
        }
    }

    pub fn apply_to_mat(&self, rhs: &Mat3) -> Mat3 {
        Self::new_from_columns(
            &self.apply_to_vec(&rhs.get_column(0)),
            &self.apply_to_vec(&rhs.get_column(1)),
            &self.apply_to_vec(&rhs.get_column(2))
        )
    }


    pub fn det(&self) -> f32 {
        let s = &self.d;
        let d0 = s[0] * {
            s[1 * 3 + 1] * s[2 * 3 + 2] -
            s[1 * 3 + 2] * s[2 * 3 + 1]
        };
        let d1 = s[1] * {
            s[1 * 3 + 0] * s[2 * 3 + 2] -
            s[1 * 3 + 2] * s[2 * 3 + 0]
        };
        let d2 = s[2] * {
            s[1 * 3 + 0] * s[2 * 3 + 1] -
            s[1 * 3 + 1] * s[2 * 3 + 0]
        };

        d0 - d1 + d2
    }

    pub fn inverse(&self) -> Option<Mat3> {
        let det = self.det();
        if det == 0.0 {
            return None
        };

        let m = &self.d;
        let invdet = 1.0/det;

        // adopted from https://stackoverflow.com/a/18504573/7487237
        Some(Mat3 { d:[
            (m[1 * 3 + 1] * m[2 * 3 + 2] - m[2 * 3 + 1] * m[1 * 3 + 2]) * invdet,
            (m[0 * 3 + 2] * m[2 * 3 + 1] - m[0 * 3 + 1] * m[2 * 3 + 2]) * invdet,
            (m[0 * 3 + 1] * m[1 * 3 + 2] - m[0 * 3 + 2] * m[1 * 3 + 1]) * invdet,
            (m[1 * 3 + 2] * m[2 * 3 + 0] - m[1 * 3 + 0] * m[2 * 3 + 2]) * invdet,
            (m[0 * 3 + 0] * m[2 * 3 + 2] - m[0 * 3 + 2] * m[2 * 3 + 0]) * invdet,
            (m[1 * 3 + 0] * m[0 * 3 + 2] - m[0 * 3 + 0] * m[1 * 3 + 2]) * invdet,
            (m[1 * 3 + 0] * m[2 * 3 + 1] - m[2 * 3 + 0] * m[1 * 3 + 1]) * invdet,
            (m[2 * 3 + 0] * m[0 * 3 + 1] - m[0 * 3 + 0] * m[2 * 3 + 1]) * invdet,
            (m[0 * 3 + 0] * m[1 * 3 + 1] - m[1 * 3 + 0] * m[0 * 3 + 1]) * invdet
        ]})

    }

}


#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct VertexIndexed {
    pub pos: u32,
    pub norm: Option<u32>,
    pub tex: Option<u32>
}

struct VertexDeindex<'a> {
    pub pos: &'a Vec3,
    pub norm: Option<&'a Vec3>,
    pub tex: Option<&'a TextureCoord>
}

#[derive(Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub tex: TextureCoord
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct GlVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex: [f32; 2]
}

impl From<Vertex> for GlVertex {
    fn from(value: Vertex) -> Self {
        GlVertex { 
            position: value.pos.into(), 
            normal: value.normal.into(), 
            tex: value.tex.into() 
        }
    }
}

pub struct MeshDataBuffs {
    pub verts: Vec<GlVertex>,
    pub indices: Vec<u32>
}

impl From<MeshData> for MeshDataBuffs {
    fn from(value: MeshData) -> Self {
        let mut added: HashMap<&VertexIndexed, usize> = Default::default();
        let mut ret: Self = Self{verts: vec![], indices: vec![]};
        for (i, vert) in value.f.iter().enumerate() {
            match added.get(&vert) {
                Some(&idx) => ret.indices.push(idx as u32),
                None => {
                    added.insert(&vert, ret.verts.len());
                    ret.indices.push(ret.verts.len() as u32);
                    ret.verts.push(value.get_vertex(i)
                        .expect("indexed wrong in enumeration")
                        .into());
                }
            };
        };

        return ret;
    }
}

pub struct MeshData {
    v: Vec<Vec3>,
    vn: Vec<Vec3>,
    vt: Vec<TextureCoord>,
    f: Vec<VertexIndexed>
}

#[derive(Debug)]
pub enum MeshError {
    VertexPositionIndexInvalid {tried: u32, max: u32},
    VertexNormalIndexInvalid {tried: u32, max: u32},
    VertexTextureIndexInvalid {tried: u32, max: u32},
    TriangleIndexInvalid {tried: u32, max: u32},
    TriangleVertexIndexInvalid {tried: u32}
}

impl MeshData {
    pub fn new() -> Self {
        MeshData { 
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


    /// add a single vertex to the index buffer
    /// use of the function is discuraged because 
    /// in the case of failure part way through 
    /// adding a polygon, the buffer will have been
    /// already updated
    pub fn add_tri_vertex(&mut self, vtx: VertexIndexed) -> Result<(), MeshError> {
        let _ = self.deref_vertex(&vtx)?;

        self.f.push(vtx);

        Ok(())
    }

    /// adds a tri to the index buffer
    pub fn add_tri(&mut self, vtxs: [VertexIndexed; 3]) -> Result<usize, MeshError> {
        // this is our validation, if the triangle is invalid, we don't want 
        // to have to attempt state rollback
        let _ = vtxs.iter()
            .map(|vtx| self.deref_vertex(vtx))
            .collect::<Result<Vec<_>,_>>()?;

        self.f.extend(vtxs.iter());
        Ok(self.f.len() / 3 - 1)
    }

    fn deref_tri_vtx(&self, tri_idx: usize, tri_vtx_idx: u8) -> Result<VertexDeindex, MeshError> {
        if tri_vtx_idx > 2 {
            return Err(MeshError::TriangleVertexIndexInvalid { tried: tri_vtx_idx as u32})
        };

        let vtx_idx = tri_idx * 3 + tri_vtx_idx as usize;
        self.deref_vertex(self.f.get(vtx_idx)
            .ok_or_else(|| MeshError::TriangleIndexInvalid { 
                tried: tri_idx as u32,
                max: self.f.len() as u32 / 3 
            })?)
    }

    fn verticies_from_tri_idx(&self, idx: usize) -> Result<[VertexDeindex; 3], MeshError> {
        Ok([
            self.deref_tri_vtx(idx, 0)?,
            self.deref_tri_vtx(idx, 1)?,
            self.deref_tri_vtx(idx, 2)?
        ])
    }

    /// calculates the normal for the triangle at tri_idx
    /// assumes a clockwise winding order
    fn calculate_tri_normal(&self, tri_idx: usize) -> Result<Vec3, MeshError> {
        let vtxs = self.verticies_from_tri_idx(tri_idx)?;
        let edge1 = vtxs[0].pos.sub(vtxs[1].pos);
        let edge2 = vtxs[0].pos.sub(vtxs[2].pos);
        Ok(edge1.cross(&edge2))
    }


    pub fn get_vertex(&self, idx: usize) -> Result<Vertex, MeshError> {
        let tri_idx = idx / 3;
        let tri_vertices = self.verticies_from_tri_idx(tri_idx)?;
        let p = &tri_vertices[idx % 3];
        // let p1 = &tri_vertices[(idx + 1) % 3];
        // let p2 = &tri_vertices[(idx + 2) % 3];


        let normal = match p.norm {
            None => self.calculate_tri_normal(tri_idx)?,
            Some(n) => *n
        };

        Ok(
            Vertex { 
                pos: *p.pos,
                normal,
                tex: *p.tex.unwrap_or(&TextureCoord{u: 0.0, v: 0.0})
            }
        )
    }

    pub fn edges(&self) -> MeshEdgeIterator<'_> {
        MeshEdgeIterator { 
            mesh: &self, 
            tri_index: 0, 
            vtx_index: 0 
        }
    }

    pub fn tris(&self) -> MeshTriIterator<'_> {
        MeshTriIterator { 
            mesh: &self, 
            tri_index: 0 
        }
    }
}

pub struct MeshEdgeIterator<'a> {
    mesh: &'a MeshData,
    tri_index: usize,
    vtx_index: usize
}

impl<'a> Iterator for MeshEdgeIterator<'a> {
    type Item = Edge;
    fn next(&mut self) -> Option<Self::Item> {
        if 3 * self.tri_index >= self.mesh.f.len() {
            return None;
        };
        let ret = Edge {
            start: self.mesh.get_vertex(3 * self.tri_index + self.vtx_index).unwrap(),
            end: self.mesh.get_vertex(3 * self.tri_index + ((self.vtx_index  + 1 ) % 3)).unwrap(),
        };

        self.vtx_index = (self.vtx_index + 1) % 3;
        if self.vtx_index == 0 {
            self.tri_index += 1;
        }

        Some(ret)
    }
}

#[derive(Debug)]
pub struct Edge {
    pub start: Vertex,
    pub end: Vertex
}

#[derive(Debug)]
pub struct Tri {
    v: [Vertex; 3]
}

pub struct MeshTriIterator<'a> {
    mesh: &'a MeshData,
    tri_index: usize,
}

impl<'a> Iterator for MeshTriIterator<'a> {
    type Item = Tri;
    fn next(&mut self) -> Option<Self::Item> {
        if self.tri_index >= self.mesh.f.len() / 3 {
            return None;
        };

        let ret = (0..3).map(|i| self.mesh.get_vertex(3 * self.tri_index + i).expect("valid index")).collect::<Vec<_>>().try_into().expect("valid tri").into();

        self.tri_index += 1;

        Some(ret)
    }
}

struct MeshVertexIterator<'a> {
    mesh: &'a MeshData,
    idx: usize
}

impl<'a> Iterator for MeshVertexIterator<'a> {
    type Item = Vertex;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.mesh.f.len() {
            return None;
        };

        let ret = self.mesh.get_vertex(self.idx).unwrap();

        self.idx += 1;

        Some(ret)
    }
}
