#![allow(dead_code)]

use std::{io::BufRead, convert::TryInto, ops::Sub};

use crate::mesh;


#[derive(Debug)]
enum ObjEntry {
    Vertex([f32; 3]),
    VertexNormal([f32; 3]),
    MapTexture([f32; 2]),
    Face(FaceEntry)
}

#[derive(Debug)]
enum FaceEntry {
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
    if tokens.is_empty() {
        return None;
    };

    match tokens[0] {
        "f" => parse_face(&tokens),
        "v" => Some(ObjEntry::Vertex(parse_vec3(&tokens)?)),
        "vn" => Some(ObjEntry::VertexNormal(parse_vec3(&tokens)?)),
        "vt" => Some(ObjEntry::MapTexture(parse_texture_coords(&tokens)?)),
        _ => None
    }
}

fn parse_vec3(tokens: &[&str]) -> Option<[f32; 3]> {
    // first index is the type - ignored by this function
    tokens.into_iter().skip(1).map(|t| t.parse().ok()).collect::<Option<Vec<f32>>>().map(|v| v.try_into().ok()).flatten()
}

fn parse_texture_coords(tokens: &[&str]) -> Option<[f32; 2]> {
    // first index is the type - ignored by this function
    tokens.into_iter().skip(1).map(|t| t.parse().ok()).collect::<Option<Vec<f32>>>().map(|v| v.try_into().ok()).flatten()
}

fn parse_face(tokens: &[&str]) -> Option<ObjEntry> {
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
        VertexDataSpecified::Pos => Some(ObjEntry::Face(FaceEntry::TriP(values.try_into().expect("number of attributes got messed up")))),
        VertexDataSpecified::PosTex => Some(ObjEntry::Face(FaceEntry::TriPT(values.try_into().expect("number of attributes got messed up")))),
        VertexDataSpecified::PosNorm => Some(ObjEntry::Face(FaceEntry::TriPN(values.try_into().expect("number of attributes got messed up")))),
        VertexDataSpecified::PosTexNorm => Some(ObjEntry::Face(FaceEntry::TriPTN(values.try_into().expect("number of attributes got messed up")))),
    }
}

fn calculate_normal(v: [&mesh::Vec3; 3]) -> mesh::Vec3 {
    let edge1 = v[0].sub(v[1]);
    let edge2 = v[1].sub(v[2]);
    edge1.cross(&edge2).normalize()
}


impl FaceEntry {
    fn norm(&self) -> Option<[u32; 3]> {
        match self {
            FaceEntry::TriPN(x) => Some([x[1], x[3], x[5]]),
            FaceEntry::TriPTN(x) => Some([x[2], x[5], x[8]]),
            _ => None
        }
    }

    fn tex(&self) -> Option<[u32; 3]> {
        match self {
            FaceEntry::TriPT(x) => Some([x[1], x[3], x[5]]),
            FaceEntry::TriPTN(x) => Some([x[1], x[4], x[7]]),
            _ => None
        }
    }

    fn pos(&self) -> [u32; 3] {
        match self {
            FaceEntry::TriP(x) => *x,
            FaceEntry::TriPT(x) => [x[0], x[2], x[4]],
            FaceEntry::TriPN(x) => [x[0], x[2], x[4]],
            FaceEntry::TriPTN(x) => [x[0], x[3], x[6]],
        }
    }

    fn build_vtx(&self) -> [mesh::VertexIndexed; 3] {
        let mut vtxs: [mesh::VertexIndexed; 3] = Default::default();
        self.norm().map(|n| n.iter().zip(vtxs.iter_mut()).for_each(|(n, v)| v.norm = Some(*n)));
        self.pos().iter().zip(vtxs.iter_mut()).for_each(|(p, v)| v.pos = *p);
        self.tex().map(|t| t.iter().zip(vtxs.iter_mut()).for_each(|(t, v)| v.tex = Some(*t)));
        vtxs
    }
}

fn handle_objentry(o: ObjEntry, m: &mut mesh::MeshData) -> Result<(), mesh::MeshError> {
    match o {
        ObjEntry::Vertex(v) => m.add_vertex_pos(v.into()),
        ObjEntry::VertexNormal(vn) => m.add_vertex_normal(vn.into()),
        ObjEntry::MapTexture(vt) => m.add_vertex_uv(vt.into()),
        ObjEntry::Face(f) => m.add_tri(f.build_vtx())?
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


