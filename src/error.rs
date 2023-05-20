use std::error::Error;
use std::fmt::Display;

#[derive(Debug)]
pub enum ObjError {
    NormalizeZeroVector
}

impl Display for ObjError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ObjError::NormalizeZeroVector => "Tried to normalize a zero magnitude vector",
        })
    }
}

impl Error for ObjError {}
