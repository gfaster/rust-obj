use std::convert::TryInto;
use std::error::Error;
use std::fmt::Display;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::collections::HashMap;

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
}

impl Display for WavefrontObjError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for WavefrontObjError {}

type ObjResult<T> = Result<T, Box<dyn Error>>;

fn read_line(line: &str, obj: &mut mesh::MeshData, path: impl AsRef<Path>, mtl_registry: &mut HashMap<String, Material>) -> ObjResult<()> {
    let noncomment = match line.split('#').next() {
        None => line,
        Some(s) => s,
    };

    let tokens: Vec<&str> = noncomment.split_whitespace().collect();
    if tokens.is_empty() {
        return Ok(());
    };

    let mut curr_mat = Material::new_dev();

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
            Material::load(path.as_ref().with_file_name(tokens.get(1).ok_or(WavefrontObjError::MissingArguments)?), mtl_registry);
        }
        "usemtl" => {
            match mtl_registry.get(*tokens.get(1).ok_or(WavefrontObjError::MissingArguments)?) {
                None => return Err(WavefrontObjError::UnknownMaterial.into()),
                Some(mat) => {
                    curr_mat = *mat;
                    obj.add_material(curr_mat);
                },
            }
        }
        _ => (),
    };
    Ok(())
}

fn parse_vec3(tokens: &[&str]) -> ObjResult<Vec3> {
    // first index is the type - ignored by this function
    // oops - I accidentally made this weirdly complicated
    tokens
        .iter()
        .skip(1)
        .take(3)
        .map(|t| t.parse())
        .collect::<Result<Vec<f32>, _>>()
        .map_err(|e| Into::<Box<dyn Error>>::into(Box::new(e)))
        .map(|v| {
            TryInto::<[f32; 3]>::try_into(v)
                .map_err(|_| WavefrontObjError::InvalidVectorFormat.into())
        })?
        .map(Into::into)
}

fn parse_texture_coords(tokens: &[&str]) -> ObjResult<Vec2> {
    // first index is the type - ignored by this function
    if !matches!(tokens.len(), 3 | 4) {
        return Err(WavefrontObjError::InvalidTexturePosFormat.into());
    }
    let mut res: ObjResult<Vec2> = tokens
        .iter()
        .skip(1)
        .take(2)
        .map(|t| t.parse())
        .collect::<Result<Vec<f32>, _>>()
        .map_err(|e| Into::<Box<dyn Error>>::into(Box::new(e)))
        .map(|v| {
            TryInto::<[f32; 2]>::try_into(v)
                .map_err(|_| WavefrontObjError::InvalidTexturePosFormat.into())
        })?
        .map(Into::into);

    res.as_mut()
        .map(|res| {
            res.x = 1.0 - res.x;
        })
        .unwrap_or(());
    res
}

fn parse_face(tokens: &[&str]) -> ObjResult<[VertexIndexed; 3]> {
    // I want this to only parse triangles for now
    if tokens.len() != 4 {
        return Err(WavefrontObjError::TooManyFaceVertices.into());
    };

    // TODO: fewer allocations here
    let attrs: Vec<_> = tokens[1..]
        .iter()
        .map(|t: &&str| {
            let mut v = t.split('/').map(|sv| sv.parse::<u32>().ok().map(|i| i - 1));

            let pos = v.next()??;
            let tex = v.next().flatten();
            let norm = v.next().flatten();
            Some(VertexIndexed { pos, tex, norm })
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(WavefrontObjError::InvalidFaceFormat)?;

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

    while buf.read_line(&mut line).map_or(0, |x| x) != 0 {
        read_line(line.as_str(), &mut objmesh, &path, &mut mtl_registry).unwrap_or_else(|e| {
            eprintln!("{:?} could not be read with error {e}", line);
        });
        line.clear();
    }

    Ok(objmesh)
}
