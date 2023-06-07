use std::fmt::Display;

pub mod consts;

/// color with u8 components
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// compile-time convert to array
    pub const fn to_array(self) -> [u8; 4] {
        // SAFTEY: Color has repr(C) and so has array layout
        unsafe {
            std::mem::transmute(self)
        }
    }

    /// compile-time create from array
    pub const fn from_array(arr: [u8; 4]) -> Self {
        // SAFTEY: Color has repr(C) and so has array layout
        unsafe {
            std::mem::transmute(arr)
        }
    }
}

impl Default for Color {
    fn default() -> Self {
        consts::GREY
    }
}


impl Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }
}

impl From<[u8; 3]> for Color {
    fn from(value: [u8; 3]) -> Self {
        Color {
            r: value[0],
            g: value[1],
            b: value[2],
            a: 255,
        }
    }
}

impl From<Color> for [u8; 3] {
    fn from(value: Color) -> Self {
        [value.r, value.g, value.b]
    }
}

impl From<[u8; 4]> for Color {
    fn from(value: [u8; 4]) -> Self {
        Color {
            r: value[0],
            g: value[1],
            b: value[2],
            a: value[3],
        }
    }
}

impl From<Color> for [u8; 4] {
    fn from(value: Color) -> Self {
        [value.r, value.g, value.b, value.a]
    }
}

impl From<(u8, u8, u8, u8)> for Color {
    fn from(value: (u8, u8, u8, u8)) -> Self {
        Color {
            r: value.0,
            g: value.1,
            b: value.2,
            a: value.3,
        }
    }
}

impl From<Color> for (u8, u8, u8, u8) {
    fn from(value: Color) -> Self {
        (value.r, value.g, value.b, value.a)
    }
}

impl From<ColorFloat> for Color {
    fn from(value: ColorFloat) -> Self {
        Color { r: value.r as u8, g: value.g as u8, b: value.b as u8, a: value.a as u8 }
    }
}

impl From<Color> for ColorFloat {
    fn from(value: Color) -> Self {
        ColorFloat { r: value.r as f32, g: value.g as f32, b: value.b as f32, a: value.a as f32 }
    }
}



/// color with rgb component range between 0.0 and 1.0
#[derive(Debug, Clone, Copy)]
pub struct ColorFloat {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}


impl Display for ColorFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let comps: [u8; 3] = (*self).into();
        write!(f, "#{:02X}{:02X}{:02X}", comps[0], comps[1], comps[2])
    }
}

impl Default for ColorFloat {
    fn default() -> Self {
        consts::GREY.into()
    }
}


impl From<[f32; 3]> for ColorFloat {
    fn from(value: [f32; 3]) -> Self {
        Self { r: value[0], g: value[1], b: value[2], a: 1.0 }
    }
}

impl From<ColorFloat> for [f32; 3] {
    fn from(value: ColorFloat) -> Self {
        [value.r, value.g, value.b]
    }
}


impl From<[u8; 3]> for ColorFloat {
    fn from(value: [u8; 3]) -> Self {
        Self { r: value[0] as f32 / 256.0, g: value[1] as f32 / 256.0, b: value[2] as f32 / 256.0, a: 1.0 }
    }
}

impl From<ColorFloat> for [u8; 3] {
    fn from(value: ColorFloat) -> Self {
        [(value.r * 256.0) as u8, (value.g * 256.0) as u8, (value.b * 256.0) as u8]
    }
}

impl From<[f32; 4]> for ColorFloat {
    fn from(value: [f32; 4]) -> Self {
        Self { r: value[0], g: value[1], b: value[2], a: value[3] }
    }
}

impl From<ColorFloat> for [f32; 4] {
    fn from(value: ColorFloat) -> Self {
        [value.r, value.g, value.b, value.a]
    }
}


impl From<[u8; 4]> for ColorFloat {
    fn from(value: [u8; 4]) -> Self {
        Self { r: value[0] as f32 / 256.0, g: value[1] as f32 / 256.0, b: value[2] as f32 / 256.0, a: value[3] as f32 / 256.0 }
    }
}

impl From<ColorFloat> for [u8; 4] {
    fn from(value: ColorFloat) -> Self {
        [(value.r * 256.0) as u8, (value.g * 256.0) as u8, (value.b * 256.0) as u8, (value.a * 256.0) as u8]
    }
}

