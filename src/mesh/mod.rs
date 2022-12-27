

#[derive(Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

#[derive(Debug)]
pub struct TextureCoord {
    pub u: f32,
    pub v: f32
}

pub struct Vertex<'a> {
    pub position: &'a Vec3,
    pub normal: Option<&'a Vec3>,
    pub binormal: Option<&'a Vec3>,
    pub tangent: Option<&'a Vec3>,
    pub albedo: Option<&'a TextureCoord>,
}

/// This isn't actually an index buffer to take advantage of type safety, but it 
/// formatted in effectively the same way
pub struct IndexBuffer<'a> {
    buffer: &'a Vec<&'a Vertex<'a>>
}

