#[allow(unused_macros)]
macro_rules! log {
    ($($tok:tt)*) => {
        eprintln!(
            "[{}]: {}",
            module_path!().rsplit_once("::").unwrap().1,
            format!($($tok)*)
        )
    };
    () => {
        eprintln!(
            "[{}]: Line {}",
            module_path!().rsplit_once("::").unwrap().1,
            line!()
        )
    };
}
