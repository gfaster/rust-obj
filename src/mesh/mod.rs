#![allow(dead_code)]

use std::{collections::HashMap, sync::Arc};

pub mod color;
pub mod mtl;
pub mod primitive;
mod tri;

pub use glm::{Vec2, Vec3};
pub use tri::*;

use self::mtl::Material;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Default)]
pub struct VertexIndexed {
    pub pos: u32,
    pub norm: Option<u32>,
    pub tex: Option<u32>,
}

struct VertexDeindex<'a> {
    pub pos: &'a Vec3,
    pub norm: Option<&'a Vec3>,
    pub tex: Option<&'a Vec2>,
}

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub tex: Vec2,
}

#[derive(Clone, Copy, Debug)]
pub struct GlVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex: [f32; 2],
}

impl From<Vertex> for GlVertex {
    fn from(value: Vertex) -> Self {
        GlVertex {
            position: value.pos.into(),
            normal: value.normal.into(),
            tex: value.tex.into(),
        }
    }
}

pub struct MeshDataBuffs<Vtx>
where
    Vtx: std::convert::From<Vertex>,
{
    pub verts: Vec<Vtx>,
    pub indices: Vec<u32>,
    pub numbering: Vec<u32>,
}

impl<Vtx> From<MeshData> for MeshDataBuffs<Vtx>
where
    Vtx: std::convert::From<Vertex>,
{
    fn from(value: MeshData) -> Self {
        let mut added: HashMap<&VertexIndexed, usize> = Default::default();
        let mut ret: Self = Self {
            verts: vec![],
            indices: vec![],
            numbering: vec![],
        };
        for (i, vert) in value.f.iter().enumerate() {
            if i % 3 == 0 {
                ret.numbering.push(ret.numbering.len() as u32);
            }
            match added.get(&vert) {
                Some(&idx) => ret.indices.push(idx as u32),
                None => {
                    added.insert(vert, ret.verts.len());
                    ret.indices.push(ret.verts.len() as u32);
                    ret.verts.push(
                        value
                            .get_vertex(i)
                            .expect("indexed wrong in enumeration")
                            .into(),
                    );
                }
            };
        }

        ret
    }
}

impl MeshData {
    pub fn to_tri_list<Vtx>(&self) -> Vec<Vtx>
    where
        Vtx: std::convert::From<Vertex> + Clone,
    {
        let mut ret = Vec::new();
        for tri in self.tris() {
            let tri_slice: [Vtx; 3] = tri.into();
            ret.extend_from_slice(&tri_slice)
        }
        ret
    }
}

pub struct MeshData {
    v: Vec<glm::Vec3>,
    vn: Vec<glm::Vec3>,
    vt: Vec<glm::Vec2>,
    f: Vec<VertexIndexed>,

    /// exclusive upper bound of each material
    materials: Vec<(u32, Material)>,

    running_center: Vec3,
    running_volume: f32,
}

#[derive(Clone)]
pub struct MeshMeta {
    pub centroid: Vec3,
    pub normalize_factor: f32,

    /// tuple of exclusive upper bound of triangle index that the other element of tuple uses
    ///
    /// this is an [`Arc`] since we clone this somewhat often
    pub materials: Arc<[(u32, Material)]>,
}

#[derive(Debug)]
pub enum MeshError {
    VertexPositionIndexInvalid { tried: u32, max: u32 },
    VertexNormalIndexInvalid { tried: u32, max: u32 },
    VertexTextureIndexInvalid { tried: u32, max: u32 },
    TriangleIndexInvalid { tried: u32, max: u32 },
    TriangleVertexIndexInvalid { tried: u32 },
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for MeshError {}

impl MeshData {
    pub fn new() -> Self {
        MeshData {
            v: vec![],
            vn: vec![],
            vt: vec![],
            f: vec![],

            materials: Vec::new(),

            running_center: Vec3::from([0.0, 0.0, 0.0]),
            running_volume: 0.0,
        }
    }

    /// makes a copy of the mesh metadata so the original can be converted to databuffs
    /// non-destructively
    pub fn get_meta(&self) -> MeshMeta {
        MeshMeta {
            centroid: self.centroid(),
            normalize_factor: self.normalize_factor(),
            materials: self.materials.clone().into(),
        }
    }

    /// extracts the metadata of the mesh, but takes ownership of material, replacing it with a
    /// default material. This should be more performant as it reduces copying
    pub fn extract_meta(&mut self) -> MeshMeta {
        MeshMeta {
            centroid: self.centroid(),
            normalize_factor: self.normalize_factor(),
            materials: std::mem::take(&mut self.materials).into(),
        }
    }

    /// adds a new vertex position, and returns the index
    /// this function is unaware of duplicates - it's up
    /// to the caller to be efficent with memory
    pub fn add_vertex_pos(&mut self, pos: Vec3) -> usize {
        self.v.push(pos);
        self.v.len() - 1
    }

    pub fn normalize_factor(&self) -> f32 {
        let centroid = self.centroid();
        self.v
            .iter()
            .map(|v| (v - &centroid).magnitude())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0)
    }

    pub fn centroid(&self) -> Vec3 {
        self.running_center / self.running_volume
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
    pub fn add_vertex_uv(&mut self, uv: Vec2) -> usize {
        self.vt.push(uv);
        self.vt.len() - 1
    }

    pub fn tri_cnt(&self) -> usize {
        self.f.len() / 3
    }

    /// gets references to the actual Vec3 comprising verticies
    /// returns a MeshError if any index is out of bounds
    fn deref_vertex(&self, vtx: &VertexIndexed) -> Result<VertexDeindex, MeshError> {
        Ok(VertexDeindex {
            pos: self.v.get(vtx.pos as usize).ok_or({
                MeshError::VertexPositionIndexInvalid {
                    tried: vtx.pos,
                    max: (self.v.len() - 1) as u32,
                }
            })?,
            norm: match vtx.norm {
                Some(vn) => Some(self.vn.get(vn as usize).ok_or({
                    MeshError::VertexNormalIndexInvalid {
                        tried: vn,
                        max: (self.vn.len() - 1) as u32,
                    }
                })?),
                None => None,
            },
            tex: match vtx.tex {
                Some(vt) => Some(self.vt.get(vt as usize).ok_or({
                    MeshError::VertexTextureIndexInvalid {
                        tried: vt,
                        max: (self.vt.len() - 1) as u32,
                    }
                })?),
                None => None,
            },
        })
    }

    /// gets the materials used in the form of exclusive upper bounds
    pub fn materials(&self) -> &[(u32, Material)] {
        &self.materials
    }

    pub fn material_mut(&mut self) -> &mut [(u32, Material)] {
        &mut self.materials
    }

    pub fn set_material(&mut self, mat: Material) {
        self.materials.push(((self.f.len() / 3) as u32, mat));
    }

    /// adds a tri to the index buffer
    pub fn add_tri(&mut self, vtxs: [VertexIndexed; 3]) -> Result<usize, MeshError> {
        // this is our validation, if the triangle is invalid, we don't want
        // to have to attempt state rollback.
        // I don't like that this requires an allocation though
        let vd = vtxs
            .iter()
            .map(|vtx| self.deref_vertex(vtx))
            .collect::<Result<Vec<_>, _>>()?;

        // https://stackoverflow.com/a/67078389
        // contribute to the centroid of the mesh
        let center = (vd[0].pos + vd[1].pos + vd[2].pos) / 4.0;
        let volume = vd[0].pos.dot(&vd[1].pos.cross(vd[2].pos)) / 6.0;
        self.running_center += center * volume;
        self.running_volume += volume;

        self.f.extend(vtxs.iter());
        Ok(self.f.len() / 3 - 1)
    }

    fn deref_tri_vtx(&self, tri_idx: usize, tri_vtx_idx: u8) -> Result<VertexDeindex, MeshError> {
        if tri_vtx_idx > 2 {
            return Err(MeshError::TriangleVertexIndexInvalid {
                tried: tri_vtx_idx as u32,
            });
        };

        let vtx_idx = tri_idx * 3 + tri_vtx_idx as usize;
        self.deref_vertex(self.f.get(vtx_idx).ok_or(MeshError::TriangleIndexInvalid {
            tried: tri_idx as u32,
            max: self.f.len() as u32 / 3,
        })?)
    }

    fn verticies_from_tri_idx(&self, idx: usize) -> Result<[VertexDeindex; 3], MeshError> {
        Ok([
            self.deref_tri_vtx(idx, 0)?,
            self.deref_tri_vtx(idx, 1)?,
            self.deref_tri_vtx(idx, 2)?,
        ])
    }

    /// calculates the normal for the triangle at tri_idx
    /// assumes a clockwise winding order
    fn calculate_tri_normal(&self, tri_idx: usize) -> Result<Vec3, MeshError> {
        // log!("inferring normal");
        let vtxs = self.verticies_from_tri_idx(tri_idx)?;
        let edge1 = vtxs[0].pos - vtxs[1].pos;
        let edge2 = vtxs[0].pos - vtxs[2].pos;
        Ok(edge1.cross(&edge2).normalize())
    }

    pub fn get_vertex(&self, idx: usize) -> Result<Vertex, MeshError> {
        let tri_idx = idx / 3;
        let tri_vertices = self.verticies_from_tri_idx(tri_idx)?;
        let p = &tri_vertices[idx % 3];
        // let p1 = &tri_vertices[(idx + 1) % 3];
        // let p2 = &tri_vertices[(idx + 2) % 3];

        let normal = match p.norm {
            None => self.calculate_tri_normal(tri_idx)?,
            Some(n) => *n,
        };

        Ok(Vertex {
            pos: *p.pos,
            normal,
            tex: *p.tex.unwrap_or(&Vec2::from([0.0f32, 0.0f32])),
        })
    }

    pub fn edges(&self) -> MeshEdgeIterator<'_> {
        MeshEdgeIterator {
            mesh: self,
            tri_index: 0,
            vtx_index: 0,
        }
    }

    pub fn tris(&self) -> MeshTriIterator<'_> {
        MeshTriIterator {
            mesh: self,
            tri_index: 0,
        }
    }

    pub fn tri_indices(&self) -> impl Iterator<Item = [u32; 3]> + '_ {
        self.f
            .chunks_exact(3)
            .map(|a| [a[0].pos, a[1].pos, a[2].pos])
    }
}

pub struct MeshEdgeIterator<'a> {
    mesh: &'a MeshData,
    tri_index: usize,
    vtx_index: usize,
}

impl<'a> Iterator for MeshEdgeIterator<'a> {
    type Item = Edge;
    fn next(&mut self) -> Option<Self::Item> {
        if 3 * self.tri_index >= self.mesh.f.len() {
            return None;
        };
        let ret = Edge {
            start: self
                .mesh
                .get_vertex(3 * self.tri_index + self.vtx_index)
                .unwrap(),
            end: self
                .mesh
                .get_vertex(3 * self.tri_index + ((self.vtx_index + 1) % 3))
                .unwrap(),
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
    pub end: Vertex,
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

        let retv = (0..3)
            .map(|i| {
                self.mesh
                    .get_vertex(3 * self.tri_index + i)
                    .expect("valid index")
            })
            .collect::<Vec<Vertex>>();
        let ret = <Vec<_> as TryInto<[Vertex; 3]>>::try_into(retv)
            .expect("valid_try")
            .into();

        self.tri_index += 1;

        Some(ret)
    }
}

struct MeshVertexIterator<'a> {
    mesh: &'a MeshData,
    idx: usize,
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
