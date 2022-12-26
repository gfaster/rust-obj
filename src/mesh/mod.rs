

pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32
}

pub struct TextureCoord {
    u: f32,
    v: f32
}

struct Vertex<'a> {
    position: &'a Vec3,
    normal: Option<&'a Vec3>,
    binormal: Option<&'a Vec3>,
    tangent: Option<&'a Vec3>,
    albedo: Option<&'a TextureCoord>,
}

struct Tri<'a> {
}

struct MeshConnections<'a> {
    buffer: &'a Vec<&'a Vertex<'a>>
}

