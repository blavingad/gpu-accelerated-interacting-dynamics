"""Compile the manuscript and package its sources."""
from pathlib import Path
import argparse
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def collect_sources(paper):
    sources, visited, figures = [], set(), set()

    def collect(path):
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        sources.append(path)
        text = path.read_text()
        figures.update(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text))
        for name in re.findall(r"\\input\{([^}]+)\}", text):
            child = paper / name
            collect(child if child.suffix else child.with_suffix(".tex"))

    collect(paper / "main.tex")
    text = "\n".join(path.read_text() for path in sources)
    entries = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", text))
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\w*(?:\[[^\]]*\])*\{([^}]+)\}", text)
        for key in group.split(",")
    }
    if cited - entries:
        raise RuntimeError("Missing bibliography entries: " + ", ".join(sorted(cited - entries)))
    return sources, figures


def main(engine="tectonic"):
    executable = shutil.which(engine)
    if executable is None:
        raise SystemExit("Tectonic was not found. Install it or pass --engine /path/to/tectonic.")
    paper = ROOT / "paper"
    sources, figures = collect_sources(paper)
    result = subprocess.run([executable, "main.tex"], cwd=paper, capture_output=True, text=True)
    log = result.stdout + result.stderr
    (paper / "build.log").write_text(log)
    result.check_returncode()
    if re.search(r"undefined references|undefined citations|Reference .* undefined|Citation .* undefined|Overfull \\[hv]box", log):
        raise RuntimeError("Unresolved reference or layout overflow; see paper/build.log.")
    files = (sources + [paper / name for name in sorted(figures)]
             + sorted((paper / "evidence").glob("*.csv"))
             + sorted((paper / "evidence").glob("*.json"))
             + [ROOT / "LICENSE", ROOT / "NOTICE"])
    with zipfile.ZipFile(paper / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name if path.parent == ROOT else path.relative_to(paper))
    print("Built paper/main.pdf and paper/source.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="tectonic")
    main(parser.parse_args().engine)
