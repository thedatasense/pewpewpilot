use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        if let Ok(output) = Command::new("clang").arg("--print-runtime-dir").output() {
            if output.status.success() {
                let dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !dir.is_empty() {
                    println!("cargo:rustc-link-search=native={dir}");
                    println!("cargo:rustc-link-lib=static=clang_rt.osx");
                }
            }
        }
    }
}
