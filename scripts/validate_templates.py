from __future__ import annotations

import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPOSITORY = "ghcr.io/agrestisdavid/faster-qwen3-tts-wyoming:latest"
EXPECTED_ICON_URL = (
    "https://raw.githubusercontent.com/agrestisdavid/"
    "faster-qwen3-tts-wyoming/main/icon.png"
)
EXPECTED_TEMPLATE_URL = (
    "https://raw.githubusercontent.com/agrestisdavid/"
    "faster-qwen3-tts-wyoming/main/templates/faster-qwen3-tts-wyoming.xml"
)
REQUIRED_TEMPLATE_FIELDS = {
    "Name",
    "Repository",
    "Icon",
    "Overview",
    "Project",
    "Support",
    "TemplateURL",
}
REQUIRED_CONFIG_ATTRIBUTES = {
    "Name",
    "Target",
    "Default",
    "Type",
    "Display",
    "Required",
    "Mask",
}
PLACEHOLDERS = {"YOUR_", "example-app", "YOUR_SUPPORT_TOPIC"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def text_of(root: ET.Element, name: str) -> str:
    element = root.find(name)
    return (element.text or "").strip() if element is not None else ""


def validate_profile() -> None:
    path = ROOT / "ca_profile.xml"
    root = ET.parse(path).getroot()
    require(root.tag == "CommunityApplications", "ca_profile.xml has the wrong root")
    require(bool(text_of(root, "Profile")), "ca_profile.xml requires a Profile")
    require(text_of(root, "Icon") == EXPECTED_ICON_URL, "profile has an unexpected Icon")
    serialized = path.read_text(encoding="utf-8")
    require(not any(item in serialized for item in PLACEHOLDERS), "profile has placeholders")


def validate_template(path: Path) -> None:
    root = ET.parse(path).getroot()
    require(root.tag == "Container", f"{path.name}: root must be Container")
    require(root.attrib.get("version") == "2", f"{path.name}: template version must be 2")

    missing = sorted(field for field in REQUIRED_TEMPLATE_FIELDS if not text_of(root, field))
    require(not missing, f"{path.name}: missing fields: {', '.join(missing)}")
    require(
        text_of(root, "Repository") == EXPECTED_REPOSITORY,
        f"{path.name}: unexpected Repository",
    )
    require(
        text_of(root, "TemplateURL") == EXPECTED_TEMPLATE_URL,
        f"{path.name}: unexpected TemplateURL",
    )
    require(
        text_of(root, "Icon") == EXPECTED_ICON_URL,
        f"{path.name}: unexpected Icon",
    )
    require(text_of(root, "Privileged") == "false", f"{path.name}: must not be privileged")

    configs = root.findall("Config")
    require(bool(configs), f"{path.name}: at least one Config is required")
    seen: set[tuple[str, str]] = set()
    for config in configs:
        missing_attrs = sorted(REQUIRED_CONFIG_ATTRIBUTES - set(config.attrib))
        require(
            not missing_attrs,
            f"{path.name}: Config {config.attrib.get('Name')!r} misses "
            f"{', '.join(missing_attrs)}",
        )
        key = (config.attrib["Type"], config.attrib["Target"])
        require(key not in seen, f"{path.name}: duplicate Config target {key}")
        seen.add(key)

    serialized = path.read_text(encoding="utf-8")
    require(not any(item in serialized for item in PLACEHOLDERS), f"{path.name}: placeholders")


def validate_icons() -> None:
    ET.parse(ROOT / "icon.svg")

    png = (ROOT / "icon.png").read_bytes()
    require(png.startswith(b"\x89PNG\r\n\x1a\n"), "icon.png is not a PNG file")
    require(png[12:16] == b"IHDR", "icon.png has no IHDR header")
    width, height = struct.unpack(">II", png[16:24])
    require((width, height) == (512, 512), "icon.png must be 512x512")


def main() -> int:
    validate_profile()
    templates = sorted((ROOT / "templates").glob("*.xml"))
    require(bool(templates), "no templates found")
    for template in templates:
        validate_template(template)
    validate_icons()
    print(
        f"Validated ca_profile.xml, icon.svg, icon.png, and "
        f"{len(templates)} template(s)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ET.ParseError, ValueError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
