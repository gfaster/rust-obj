use std::{
    error::Error,
    io::{BufRead, BufReader},
    path::Path,
    sync::Arc,
};

use image::RgbaImage;

use super::color::ColorFloat;

#[derive(Clone)]
pub struct Material {
    name: String,
    diffuse: ColorFloat,
    ambient: ColorFloat,
    specular: ColorFloat,
    spec_exp: f32,

    ambient_map: Option<Arc<RgbaImage>>,
    diffuse_map: Option<Arc<RgbaImage>>,
    normal_map: Option<Arc<RgbaImage>>,
    // specular_map: Option<RgbaImage>,
}

impl Material {
    pub fn new() -> Self {
        Self {
            name: "New_Material".to_string(),
            diffuse: Default::default(),
            ambient: [0.05, 0.05, 0.05].into(),
            specular: [0.0, 0.0, 0.0].into(),
            ambient_map: Default::default(),
            diffuse_map: Default::default(),
            normal_map: Default::default(),
            spec_exp: 1.0,
        }
    }

    /// creates a new material with a dev texture
    pub fn new_dev() -> Self {
        Self {
            name: "New_Material".to_string(),
            diffuse: Default::default(),
            ambient: [0.05, 0.05, 0.05].into(),
            specular: [0.0, 0.0, 0.0].into(),
            ambient_map: Default::default(),
            diffuse_map: Some(Material::dev_texture().into()),
            normal_map: Default::default(),
            spec_exp: 1.0,
        }
    }

    pub fn dev_texture() -> RgbaImage {
        image::load_from_memory_with_format(
            include_bytes!("../../assets/missing_texture.png"),
            image::ImageFormat::Png,
        )
        .unwrap()
        .to_rgba8()
    }

    pub fn diffuse(&self) -> ColorFloat {
        self.diffuse
    }

    pub fn ambient(&self) -> ColorFloat {
        self.ambient
    }

    pub fn specular(&self) -> ColorFloat {
        self.specular
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn diffuse_map(&self) -> Option<Arc<RgbaImage>> {
        self.diffuse_map.clone()
    }

    pub fn normal_map(&self) -> Option<Arc<RgbaImage>> {
        self.normal_map.clone()
    }

    pub fn specular_map(&self) -> Option<&RgbaImage> {
        // self.specular_map.as_ref()
        None
    }

    pub fn base_specular_factor(&self) -> f32 {
        self.spec_exp
    }

    /// load an mtl file
    ///
    /// I would like to just be able to pass a reader here, but this function needs to open
    /// multiple files, probably with relative pathing
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let reader = BufReader::new(std::fs::OpenOptions::new().read(true).open(&path)?);

        let mut lines = reader.lines().collect::<std::io::Result<Vec<_>>>()?;
        for l in &mut lines {
            if let Some(split) = l.split_once('#') {
                *l = split.0.to_string()
            }
        }
        lines.retain(|l| !l.is_empty());

        let mut ret = Self::new();

        let name = match lines[0].split_whitespace().collect::<Vec<_>>()[..] {
            [] => return Err(MtlError::MissingDirective.into()),
            ["newmtl", n] => n,
            ["newmtl"] => return Err(MtlError::MissingArgument.into()),
            [invalid_directive, ..] => {
                return Err(MtlError::InvalidDirective(invalid_directive.to_string()).into())
            }
        };
        ret.name = name.to_string();

        for line in &lines[1..] {
            let tokens = &line.split_whitespace().collect::<Vec<_>>()[..];

            match tokens {
                [] => return Err(MtlError::UnexpectedEol.into()),
                ["newmtl", ..] => return Err(MtlError::MultipleMaterials.into()),
                ["Ka", r, g, b] => ret.ambient = [r.parse::<f32>()?, g.parse()?, b.parse()?].into(),
                ["Kd", r, g, b] => ret.diffuse = [r.parse::<f32>()?, g.parse()?, b.parse()?].into(),
                ["Ks", r, g, b] => {
                    ret.specular = [r.parse::<f32>()?, g.parse()?, b.parse()?].into()
                }
                ["Ns", factor] => ret.spec_exp = factor.parse()?,
                ["map_Ka", map_file] => {
                    ret.ambient_map = Some(read_map(path.as_ref().with_file_name(map_file))?.into())
                }
                ["map_Kd", map_file] => {
                    ret.diffuse_map = Some(read_map(path.as_ref().with_file_name(map_file))?.into())
                }
                ["bump", map_file] | ["map_Bump", map_file] => {
                    ret.normal_map = Some(read_map(path.as_ref().with_file_name(map_file))?.into())
                }
                ["map_Ks", _map_file] => (),
                ["illum", _illium] => (),
                ["Ke", _r, _g, _b] => (),
                ["Ni", _i] => (),
                ["d", _d] => (),
                _ => return Err(MtlError::InvalidDirective(line.to_string()).into()),
            }
        }
        Ok(ret)
    }
}

fn read_map(path: impl AsRef<Path>) -> Result<RgbaImage, Box<dyn Error>> {
    Ok(image::io::Reader::open(path)?
        .with_guessed_format()?
        .decode()?
        .to_rgba8())
}

impl Default for Material {
    fn default() -> Self {
        let mut ret = Self::new();
        ret.name = "Default Material".to_string();
        ret
    }
}

impl std::fmt::Debug for Material {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("Material");

        debug_struct
            .field("name", &self.name)
            .field("diffuse", &self.diffuse.to_string())
            .field("ambient", &self.ambient.to_string())
            .field("specular", &self.specular.to_string())
            .field("specular factor", &self.spec_exp);

        if let Some(m) = &self.diffuse_map {
            debug_struct.field("diffuse_map", &m.dimensions());
        }

        debug_struct.finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub enum MtlError {
    InvalidDirective(String),
    MissingDirective,
    UnexpectedEof,
    UnexpectedEol,
    CruftAtEol,
    MissingArgument,
    MultipleMaterials,
    MissingFileExtension,
}

impl std::fmt::Display for MtlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MtlError::MultipleMaterials => {
                write!(f, "{:?} (multiple materials is unimplemented)", self)
            }
            _ => write!(f, "{:?}", self),
        }
    }
}

impl std::error::Error for MtlError {}
