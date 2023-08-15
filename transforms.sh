set -e
# file $1
cargo b -r
# seq 9 12 | xargs -I{} ./target/release/obj "$1" "{}"
dirs=$(seq 9 14| xargs -I{} sh "-c" "printf \"%b/0_norm.exr \" \$(./target/release/obj $1 {})")
gimp $(echo $dirs | tr "\n" " ")
