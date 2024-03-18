use super::{MeshData, VertexIndexed};

macro_rules! vindex {
    ($idx:literal) => {
        VertexIndexed {
            pos: $idx,
            tex: Some($idx),
            norm: Some(0),
        }
    };
}

/// quad made to cover the screen
pub fn frame() -> MeshData {
    let mut frame_mesh = MeshData::new();
    frame_mesh.add_vertex_pos(glm::vec3(0.0, 0.0, 0.0));
    frame_mesh.add_vertex_pos(glm::vec3(1.0, 0.0, 0.0));
    frame_mesh.add_vertex_pos(glm::vec3(1.0, 1.0, 0.0));
    frame_mesh.add_vertex_pos(glm::vec3(0.0, 1.0, 0.0));

    frame_mesh.add_vertex_uv(glm::vec2(0.0, 0.0));
    frame_mesh.add_vertex_uv(glm::vec2(1.0, 0.0));
    frame_mesh.add_vertex_uv(glm::vec2(1.0, 1.0));
    frame_mesh.add_vertex_uv(glm::vec2(0.0, 1.0));

    frame_mesh.add_vertex_normal(glm::vec3(0.0, 0.0, 1.0));

    frame_mesh
        .add_tri([vindex!(0), vindex!(1), vindex!(3)])
        .unwrap();
    frame_mesh
        .add_tri([vindex!(3), vindex!(1), vindex!(2)])
        .unwrap();

    frame_mesh
}
