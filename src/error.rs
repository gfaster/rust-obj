use std::error::Error;
use std::fmt::Display;

#[derive(Debug)]
#[allow(dead_code)]
pub enum ObjError {}

impl Display for ObjError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("error")
    }
}

impl Error for ObjError {}
