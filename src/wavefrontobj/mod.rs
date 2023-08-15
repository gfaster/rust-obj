use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;
use std::fmt::Display;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use crate::glm::{Vec2, Vec3};
use crate::mesh::mtl::Material;
use crate::mesh::{self, VertexIndexed};

#[derive(Debug)]
enum WavefrontObjError {
    InvalidFaceFormat,
    TooManyFaceVertices,
    InvalidVectorFormat,
    InvalidTexturePosFormat,
    InvalidDirective,
    MissingArguments,
    UnknownMaterial,
    MissingMaterial,
}

impl Display for WavefrontObjError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for WavefrontObjError {}

type ObjResult<T> = Result<T, Box<dyn Error>>;

fn read_line(
    line: &str,
    obj: &mut mesh::MeshData,
    curr_mat: &mut Option<String>,
    path: impl AsRef<Path>,
    mtl_registry: &mut HashMap<String, Material>,
) -> ObjResult<()> {
    let noncomment = match line.split('#').next() {
        None => line,
        Some(s) => s,
    };

    let tokens: Vec<&str> = noncomment.split_whitespace().collect();
    if tokens.is_empty() {
        return Ok(());
    };

    match tokens[0] {
        "f" => {
            obj.add_tri(parse_face(&tokens)?)?;
        }
        "v" => {
            obj.add_vertex_pos(parse_vec3(&tokens)?);
        }
        "vn" => {
            obj.add_vertex_normal(parse_vec3(&tokens)?);
        }
        "vt" => {
            obj.add_vertex_uv(parse_texture_coords(&tokens)?);
        }
        "mtllib" => {
            Material::load(
                path.as_ref()
                    .with_file_name(tokens.get(1).ok_or(WavefrontObjError::MissingArguments)?),
                mtl_registry,
            )
            .unwrap_or_else(|e| {log!("{e}"); log!("using default mtl"); Default::default()});
            //*curr_mat = Some(mtl_registry.keys().next().ok_or(WavefrontObjError::MissingMaterial).unwrap().clone());
        }
        "usemtl" => {
            match mtl_registry.get(*tokens.get(1).ok_or(WavefrontObjError::MissingArguments)?) {
                None => return Err(WavefrontObjError::UnknownMaterial.into()),
                Some(mat) => match curr_mat {
                    None => *curr_mat = Some(mat.name().to_string()),
                    Some(prev_mat) => {
                        obj.set_material(
                            mtl_registry
                                .get(prev_mat)
                                .expect("previous material was registered")
                                .clone(),
                        );
                        *curr_mat = Some(mat.name().to_string());
                    }
                },
            }
        }
        _ => (),
    };

    Ok(())
}

fn parse_vec3(tokens: &[&str]) -> ObjResult<Vec3> {
    // first index is the type - ignored by this function
    if tokens.len() != 4 {
        return Err(WavefrontObjError::InvalidVectorFormat.into());
    }
    return Ok(glm::vec3::<f32>(
        tokens[1].parse()?,
        tokens[2].parse()?,
        tokens[3].parse()?,
    ));
}

fn parse_texture_coords(tokens: &[&str]) -> ObjResult<Vec2> {
    // first index is the type - ignored by this function
    if !matches!(tokens.len(), 3 | 4) {
        return Err(WavefrontObjError::InvalidTexturePosFormat.into());
    }
    // need to invert y value for vulkan it seems
    return Ok(glm::vec2::<f32>(
        tokens[1].parse()?,
        1.0f32 - tokens[2].parse::<f32>()?,
    ));
}

fn parse_face(tokens: &[&str]) -> ObjResult<[VertexIndexed; 3]> {
    // I want this to only parse triangles for now
    if tokens.len() != 4 {
        return Err(WavefrontObjError::TooManyFaceVertices.into());
    };

    // made this into a for loop to avoid the allocation from collecting into a vec
    let mut attrs = [VertexIndexed::default(); 3];
    for (i, attr) in attrs.iter_mut().enumerate() {
        let mut v = tokens[i + 1]
            .split('/')
            .map(|sv| sv.parse::<u32>().ok().map(|i| i - 1));

        if let Some(pos) = v.next().flatten() {
            let tex = v.next().flatten();
            let norm = v.next().flatten();
            *attr = VertexIndexed { pos, tex, norm }
        } else {
            return Err(WavefrontObjError::InvalidFaceFormat.into());
        }
    }

    // ensure consistency of vertex attributes
    if !attrs
        .windows(2)
        .all(|x| core::mem::discriminant(&x[0].tex) == core::mem::discriminant(&x[1].tex))
    {
        return Err(WavefrontObjError::InvalidFaceFormat.into());
    }
    if !attrs
        .windows(2)
        .all(|x| core::mem::discriminant(&x[0].norm) == core::mem::discriminant(&x[1].norm))
    {
        return Err(WavefrontObjError::InvalidFaceFormat.into());
    }

    Ok(attrs
        .try_into()
        .expect("attributes should have been validated previously"))
}

pub fn load(path: impl AsRef<Path>) -> std::io::Result<mesh::MeshData> {
    let mut buf = BufReader::new(std::fs::File::open(&path)?);
    let mut line = String::new();

    let mut objmesh = mesh::MeshData::new();
    let mut mtl_registry = HashMap::new();
    let mut curr_mat = None;

    while buf.read_line(&mut line).map_or(0, |x| x) != 0 {
        read_line(
            line.as_str(),
            &mut objmesh,
            &mut curr_mat,
            &path,
            &mut mtl_registry,
        )
        .unwrap_or_else(|e| {
            eprintln!("{:?} could not be read with error {e}", line);
        });
        line.clear();
    }

    match curr_mat {
        None => objmesh.set_material(Material::new()),
        Some(mat) => objmesh.set_material(
            mtl_registry
                .get(&mat)
                .ok_or(WavefrontObjError::UnknownMaterial)
                .unwrap()
                .clone(),
        ),
    }

    log!("{:?} loaded with {:#} total triangles", path.as_ref().file_name().unwrap(), objmesh.tri_cnt());

    Ok(objmesh)
}
