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
    let start_time = std::time::Instant::now();
    log!("starting partitioning");
    let mut ne = mesh.tri_cnt() as i32;
    let mut nn = ne * 3;
    let mut eptr: Vec<_> = (0..=mesh.tri_cnt()).map(|x| (x as i32) * 3).collect();
    let mut eind: Vec<_> = mesh.tri_indices().flatten().collect();
    let mut r_xadj = std::ptr::null_mut();
    let mut r_adjncy = std::ptr::null_mut();
    unsafe {
        let res = metis::METIS_MeshToDual(
            &mut ne,
            &mut nn,
            eptr.as_mut_ptr(),
            eind.as_mut_ptr() as *mut i32,
            &mut 2,
            &mut 0,
            &mut r_xadj,
            &mut r_adjncy,
        );
        if res != metis::METIS_OK {
            if !r_xadj.is_null() {
                metis::METIS_Free(r_xadj.cast());
            }
            if !r_adjncy.is_null() {
                metis::METIS_Free(r_adjncy.cast());
            }
            return None;
        }
    }

    log!("completed preprocessing");

    assert!(!r_xadj.is_null());
    assert!(!r_adjncy.is_null());

    let mut nvtxs = ne;

    let target_size = 128;

    let mut nparts = nvtxs / target_size;

    let mut ncon = 1;

    let mut options = [-1; metis::METIS_NOPTIONS as usize];

    let mut part = vec![0; ne as usize];

    let res = unsafe {
        metis::pmetis::METIS_PartGraphRecursive(
            &mut nvtxs,
            &mut ncon,
            r_xadj,
            r_adjncy,
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

    if !r_xadj.is_null() {
        unsafe { metis::METIS_Free(r_xadj.cast()) };
    }
    if !r_adjncy.is_null() {
        unsafe { metis::METIS_Free(r_adjncy.cast()) };
    }

    if res != metis::METIS_OK {
        return None;
    }

    log!("completed partitioning in {:.3}s", (std::time::Instant::now() - start_time).as_secs_f32());

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
