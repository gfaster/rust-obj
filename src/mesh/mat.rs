use super::color::ColorFloat;

pub struct Material {
    name: String,
    diffuse: ColorFloat,
    ambient: ColorFloat,
    specular: ColorFloat,
    density: f32,

    // diffuse_map: Option<>,
}

impl Material {
    pub fn new(name: String) -> Self {
        Self {
            name,
            density: 1.0,
            ..Default::default()
        }
    }

    pub fn set_diffuse(&mut self, diffuse: ColorFloat) {
        self.diffuse = diffuse;
    }

    pub fn diffuse(&self) -> ColorFloat {
        self.diffuse
    }

    pub fn set_ambient(&mut self, ambient: ColorFloat) {
        self.ambient = ambient;
    }

    pub fn ambient(&self) -> ColorFloat {
        self.ambient
    }

    pub fn set_specular(&mut self, specular: ColorFloat) {
        self.specular = specular;
    }

    pub fn specular(&self) -> ColorFloat {
        self.specular
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new("Default Material".to_string())
    }
}
