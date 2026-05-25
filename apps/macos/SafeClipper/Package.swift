// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "safeclipper",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "safeclipper", targets: ["SafeClipperApp"])
    ],
    targets: [
        .target(name: "PrivacyFilterCore"),
        .executableTarget(
            name: "SafeClipperApp",
            dependencies: ["PrivacyFilterCore"],
            linkerSettings: [
                .unsafeFlags([
                    "-L../../../target/release",
                    "-lsafeclipper_cli",
                    "-Xlinker", "-rpath",
                    "-Xlinker", "@executable_path/../Frameworks",
                    "-Xlinker", "-rpath",
                    "-Xlinker", "../../../target/release"
                ])
            ]
        )
    ]
)
