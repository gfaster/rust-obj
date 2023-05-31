use std::{convert::TryInto, io::BufRead};

use crate::glm::{Vec2, Vec3};
use crate::mesh::{self, VertexIndexed};

fn read_line(line: &str, obj: &mut mesh::MeshData) -> Result<(), ()> {
    let noncomment = match line.splitn(2, '#').next() {
        None => line,
        Some(s) => s,
    };

    let tokens: Vec<&str> = noncomment.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(());
    };

    match tokens[0] {
        "f" => {
            obj.add_tri(parse_face(&tokens).ok_or(())?).map_err(|_| ())?;
        },
        "v" => {
            obj.add_vertex_pos(parse_vec3(&tokens).ok_or(())?.into());
        },
        "vn" => {
            obj.add_vertex_normal(parse_vec3(&tokens).ok_or(())?);
        },
        "vt" => {
            obj.add_vertex_uv(parse_texture_coords(&tokens).ok_or(())?);
        },
        _ => (),
    };
    Ok(())
}


fn parse_vec3(tokens: &[&str]) -> Option<Vec3> {
    // first index is the type - ignored by this function
    tokens
        .into_iter()
        .skip(1)
        .map(|t| t.parse().ok())
        .collect::<Option<Vec<f32>>>()
        .map(|v| TryInto::<[f32; 3]>::try_into(v).ok())
        .flatten()
        .map(Into::into)
}

fn parse_texture_coords(tokens: &[&str]) -> Option<Vec2> {
    // first index is the type - ignored by this function
    if !matches!(tokens.len(), 3 | 4) {
        return None;
    }
    tokens
        .into_iter()
        .skip(1)
        .take(2)
        .map(|t| t.parse().ok())
        .collect::<Option<Vec<f32>>>()
        .map(|v| TryInto::<[f32; 2]>::try_into(v).ok())
        .flatten()
        .map(Into::into)
}

fn parse_face(tokens: &[&str]) -> Option<[VertexIndexed; 3]> {
    // I want this to only parse triangles for now
    if tokens.len() != 4 {
        return None;
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
        .collect::<Option<Vec<_>>>()?;

    // ensure consistency of vertex attributes
    if !attrs
        .windows(2)
        .all(|x| core::mem::discriminant(&x[0].tex) == core::mem::discriminant(&x[1].tex))
    {
        return None;
    }
    if !attrs
        .windows(2)
        .all(|x| core::mem::discriminant(&x[0].norm) == core::mem::discriminant(&x[1].norm))
    {
        return None;
    }

    attrs.try_into().ok()
}

pub fn read_obj(buf: &mut impl BufRead) -> mesh::MeshData {
    let mut line = String::new();

    let mut objmesh = mesh::MeshData::new();

    while buf.read_line(&mut line).map_or(0, |x| x) != 0 {
        read_line(line.as_str(), &mut objmesh).unwrap_or_else(|_| {
            if line.splitn(2, '#').next().map_or(false, |l| l.trim().len() > 0) {
                eprintln!("Line could not be read: {:?}", line);
            }
        });
        line.clear();
    }

    objmesh
}
