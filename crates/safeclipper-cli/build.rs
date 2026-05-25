fn main() {
    #[cfg(target_os = "macos")]
    {
        cc::Build::new()
            .file("src/macos_vision_ocr.m")
            .flag("-fobjc-arc")
            .flag("-fblocks")
            .compile("safeclipper_macos_vision_ocr");

        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Vision");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rerun-if-changed=src/macos_vision_ocr.m");
    }
}
