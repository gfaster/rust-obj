use std::{
    collections::{HashMap, HashSet},
    ptr,
};

use crate::mesh::MeshData;

struct Csr {
    xadj: Vec<i32>,
    adjncy: Vec<i32>,
}

/// This is slow, but it works and I don't need to think about it very hard
fn tri_csr(mesh: &MeshData) -> Csr {
    let mut xadj = Vec::new();
    let mut adjncy = Vec::new();

    // vtx -> tri_idx(s)
    let mut vtx_map: HashMap<u32, Vec<_>> = HashMap::new();

    for (idx, tri) in mesh.tri_indices().enumerate() {
        debug_assert!(tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]);
        for vtx in tri {
            vtx_map.entry(vtx).or_default().push(idx as i32)
        }
    }

    let mut neighbors = Vec::new();
    for (idx, tri) in mesh.tri_indices().enumerate() {
        xadj.push(adjncy.len() as i32);
        for vtx in tri {
            neighbors.extend_from_slice(&vtx_map[&vtx]);
        }
        let uniq: HashSet<_> = neighbors
            .drain(..)
            .filter(|&i| i as u32 != idx as u32)
            .collect();
        adjncy.extend(uniq);
    }
    xadj.push(adjncy.len() as i32);

    Csr { xadj, adjncy }
}

pub fn partition_mesh(mesh: &MeshData) -> Option<Vec<i32>> {
    let Csr {
        mut xadj,
        mut adjncy,
    } = tri_csr(mesh);

    let mut nvtxs = xadj.len() as i32 - 1;

    let target_size = 128;

    let mut nparts = nvtxs / target_size;

    let mut ncon = 1;

    let mut options = [-1; metis::METIS_NOPTIONS as usize];

    let mut part = vec![0; xadj.len() - 1];

    let res = unsafe {
        metis::pmetis::METIS_PartGraphRecursive(
            &mut nvtxs,
            &mut ncon,
            xadj.as_mut_ptr(),
            adjncy.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut nparts,
            ptr::null_mut(),
            ptr::null_mut(),
            options.as_mut_ptr(),
            &mut 0,
            part.as_mut_ptr(),
        )
    };

    if res != metis::METIS_OK {
        return None;
    }

    Some(part)
}

pub fn partition_mesh_2(mesh: &MeshData) -> Option<Vec<i32>> {
    let Csr {
        mut xadj,
        mut adjncy,
    } = tri_csr(mesh);

    let mut nn = xadj.len() as i32 - 1;

    let mut ne = nn / 3;

    let target_size = 128;

    let mut nparts = ne / target_size;

    let mut options = [-1; metis::METIS_NOPTIONS as usize];

    let mut epart = vec![0; xadj.len() - 1];

    let mut npart = vec![0; xadj.len() - 1];

    let res = unsafe {
        metis::METIS_PartMeshDual(
            &mut ne,
            &mut nn,
            xadj.as_mut_ptr(),
            adjncy.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut 2,
            &mut nparts,
            ptr::null_mut(),
            options.as_mut_ptr(),
            ptr::null_mut(),
            epart.as_mut_ptr(),
            npart.as_mut_ptr(),
        )
    };

    if res != metis::METIS_OK {
        return None;
    }

    Some(epart)
}
