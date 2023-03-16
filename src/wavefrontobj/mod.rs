#![allow(dead_code)]

use std::{io::{self, BufRead}, convert::TryInto};

use crate::mesh;


#[derive(Debug)]
enum ObjEntry {
    Vertex(mesh::Vec3),
    VertexNormal(mesh::Vec3),
    MapTexture(mesh::TextureCoord),
    TriP([u32; 3]),
    TriPT([u32; 6]),
    TriPN([u32; 6]),
    TriPTN([u32; 9]),
}

fn read_line(line: &str) -> Option<ObjEntry> {
    let noncomment = match line.splitn(2, '#').next() {
        None => line,
        Some(s) => s
    };
    let tokens: Vec<&str> = noncomment.split_whitespace().collect();
    if tokens.len() == 0 {
        return None;
    };

    match tokens[0] {
        "f" => parse_face(tokens),
        "v" => Some(ObjEntry::Vertex(parse_vec3(tokens)?)),
        "vn" => Some(ObjEntry::VertexNormal(parse_vec3(tokens)?)),
        "vt" => Some(ObjEntry::MapTexture(parse_texture_coords(tokens)?)),
        _ => None
    }
}

fn parse_vec3(tokens: Vec<&str>) -> Option<mesh::Vec3> {
    // first index is the type - ignored by this function
    if tokens.len() != 4 { return None; };
    Some(mesh::Vec3 { 
        x: tokens[1].parse().ok()?,
        y: tokens[2].parse().ok()?,
        z: tokens[3].parse().ok()?
    })
    
}

fn parse_texture_coords(tokens: Vec<&str>) -> Option<mesh::TextureCoord> {
    // first index is the type - ignored by this function
    if tokens.len() != 3 { return None; };
    Some(mesh::TextureCoord { 
        u: tokens[1].parse().ok()?,
        v: tokens[2].parse().ok()? })
}

fn parse_face(tokens: Vec<&str>) -> Option<ObjEntry> {
    // I want this to only parse triangles for now
    if tokens.len() != 4 { return None; };
    
    // I feel like I shouldn't need this, but without it I run into later problems
    #[derive(Clone, Copy)] 
    enum VertexDataSpecified {
        Pos,
        PosTex,
        PosNorm,
        PosTexNorm
    }

    let mut types = [VertexDataSpecified::Pos; 3];
    let mut values: Vec<u32> = vec![];

    let attributes_raw = tokens[1..].iter()
        .map(|x: &&str| x.split('/'));


    // puts the line into a nice array, and finds the data it must specify
    for (i, vert) in attributes_raw.enumerate() {
        let mut attrs_vec: Vec<Option<u32>> = vert
            .map(|x| x.parse::<u32>().ok())
            .map(|x| x.map_or(None, |x| Some(x - 1)))
            .collect::<Vec<Option<u32>>>();

        attrs_vec.resize(3, None);

        let attrs: [Option<u32>; 3] = attrs_vec
            .try_into().expect("Vector should be fixed size");

        match attrs {
            [Some(_), None,    None] => types[i] = VertexDataSpecified::Pos,
            [Some(_), Some(_), None] => types[i] = VertexDataSpecified::PosTex,
            [Some(_), None,    Some(_)] => types[i] = VertexDataSpecified::PosNorm,
            [Some(_), Some(_), Some(_)] => types[i] = VertexDataSpecified::PosTexNorm,
            _ => return None
        };
        values.extend(attrs.iter().filter_map(|x| *x));
    };

    // this super elegant line checking to make sure types are the same by
    // https://github.com/bugaevc found via 
    // https://sts10.github.io/2019/06/06/is-all-equal-function.html
    // This line is why we need those derives on VertexDataSpecified
    if !types.windows(2).all(|w: &[VertexDataSpecified]| w[0] as u32 == w[1] as u32) {
        return None
    };

    match types[0] {
        VertexDataSpecified::Pos => Some(ObjEntry::TriP(values.try_into().expect("number of attributes got messed up"))),
        VertexDataSpecified::PosTex => Some(ObjEntry::TriPT(values.try_into().expect("number of attributes got messed up"))),
        VertexDataSpecified::PosNorm => Some(ObjEntry::TriPN(values.try_into().expect("number of attributes got messed up"))),
        VertexDataSpecified::PosTexNorm => Some(ObjEntry::TriPTN(values.try_into().expect("number of attributes got messed up"))),
    }
}

fn calculate_normal(v: [&mesh::Vec3; 3]) -> mesh::Vec3 {
    let edge1 = v[0].sub(&v[1]);
    let edge2 = v[1].sub(&v[2]);
    edge1.cross(&edge2).normalized().unwrap()
}

fn get_pos_from_objentry(o: &ObjEntry) -> [u32; 3] {
    match o {
        ObjEntry::TriP(x) => *x,
        ObjEntry::TriPT(x) => [x[0], x[2], x[4]],
        ObjEntry::TriPN(x) => [x[0], x[2], x[4]],
        ObjEntry::TriPTN(x) => [x[0], x[3], x[6]],
        _ => panic!("this function should only be called on face line, but it was called on {:?}", o)
    }
}

fn get_norm_from_objentry(o: &ObjEntry) -> [u32; 3] {
    match o {
        ObjEntry::TriPN(x) => [x[1], x[3], x[5]],
        ObjEntry::TriPTN(x) => [x[2], x[5], x[8]],
        _ => panic!("this function should only be called on face line with a normal, but it was called on {:?}", o)
    }
}

fn get_tex_from_objentry(o: &ObjEntry) -> [u32; 3] {
    match o {
        ObjEntry::TriPT(x) => [x[1], x[3], x[5]],
        ObjEntry::TriPTN(x) => [x[1], x[4], x[7]],
        _ => panic!("this function should only be called on face line with a uv, but it was called on {:?}", o)
    }
}

fn build_vtx_from_objentry_face(o: &ObjEntry) -> [mesh::VertexIndexed; 3]{
    let mut vtxs: Vec<mesh::VertexIndexed> = vec![];
    for i in 0..3 {
        let v = match o {
            ObjEntry::TriP(_) => {
                mesh::VertexIndexed {
                    pos: get_pos_from_objentry(o)[i],
                    norm: None,
                    tex: None
                }
            },
            ObjEntry::TriPN(_) => {
                mesh::VertexIndexed {
                    pos: get_pos_from_objentry(o)[i],
                    norm: Some(get_norm_from_objentry(o)[i]),
                    tex: None
                }
            },
            ObjEntry::TriPT(_) => {
                mesh::VertexIndexed {
                    pos: get_pos_from_objentry(o)[i],
                    norm: None,
                    tex: Some(get_tex_from_objentry(o)[i])
                }
            },
            ObjEntry::TriPTN(_) => {
                mesh::VertexIndexed {
                    pos: get_pos_from_objentry(o)[i],
                    norm: Some(get_norm_from_objentry(o)[i]),
                    tex: Some(get_tex_from_objentry(o)[i])
                }
            },
            _ => panic!("should only take faces")
        };
        vtxs.push(v);
    };

    match vtxs.try_into() {
        Err(e) => panic!("{:?}", e),
        Ok(v) => v
    }
}

fn handle_objentry(o: ObjEntry, m: &mut mesh::MeshData) -> Result<(), mesh::MeshError> {
    match o {
        ObjEntry::Vertex(v) => m.add_vertex_pos(v),
        ObjEntry::VertexNormal(vn) => m.add_vertex_normal(vn),
        ObjEntry::MapTexture(vt) => m.add_vertex_uv(vt),
        _ => m.add_tri(build_vtx_from_objentry_face(&o))?
    };
    Ok(())
}

pub fn read_obj(buf: &mut impl BufRead) -> mesh::MeshData {
    let mut line = String::new();

    let mut objmesh  = mesh::MeshData::new();
    
    while buf.read_line(&mut line).map_or(0, |x| x) != 0 {
        match read_line(line.as_str()) {
            None => (),
            Some(o) => handle_objentry(o, &mut objmesh).unwrap_or(())
        };
        line.clear();
    };

    objmesh
}


