use std::{io, convert::TryInto};
mod mesh;

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
        z: tokens[3].parse().ok()? })
    
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

    let attributes_raw = tokens[1..].iter().map(|x: &&str| x.split('/'));


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



fn main() {
    let input = io::stdin();
    let mut line = String::new();
    while input.read_line(&mut line).is_ok() {
        print!("{} => ", line);

        let res = read_line(line.as_str());

        match res {
            None => println!("---"),
            Some(x) => println!("{:?}", x)
        };

        line.clear();
    }
    
}
