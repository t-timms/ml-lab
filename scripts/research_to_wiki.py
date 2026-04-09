"""Push experiment findings to llm-wiki as knowledge pages.

Extracts key findings from RESEARCH_LOG.md and model_card.md,
formats them as wiki-compatible pages for the llm-wiki knowledge base.

Usage:
    python scripts/research_to_wiki.py --experiment-dir experiments/2026-04-gsm8k-grpo
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

WIKI_DIR = Path.home() / "Documents" / "Project Portfolio" / "llm-wiki"


def _extract_findings(research_log: str) -> list[str]:
    """Extract classified findings from RESEARCH_LOG.md.

    Looks for findings classified as Interesting, Anomalous,
    or Breakthrough-Candidate.
    """
    findings = []
    in_findings = False
    current_finding = []

    for line in research_log.split("\n"):
        if "Finding Classification" in line or "Unexpected Findings" in line:
            in_findings = True
            continue
        if in_findings and line.startswith("###"):
            if current_finding:
                findings.append("\n".join(current_finding).strip())
                current_finding = []
            in_findings = False
            continue
        if in_findings and line.strip():
            current_finding.append(line)

    if current_finding:
        findings.append("\n".join(current_finding).strip())

    return [f for f in findings if f]


def _extract_hypothesis(research_log: str) -> str:
    """Extract the hypothesis section from RESEARCH_LOG.md."""
    in_hypothesis = False
    lines = []

    for line in research_log.split("\n"):
        if re.match(r"^##\s+Hypothesis", line):
            in_hypothesis = True
            continue
        if in_hypothesis and line.startswith("##"):
            break
        if in_hypothesis:
            lines.append(line)

    text = "\n".join(lines).strip()
    if text.startswith("["):
        return ""
    return text


def generate_wiki_page(experiment_dir: Path) -> str:
    """Generate a wiki-compatible page from experiment artifacts.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Wiki page content as markdown string
    """
    experiment_name = experiment_dir.name
    date = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    sections = [f"# {experiment_name}", f"\n*Generated from ml-lab experiment on {date}*\n"]

    # Hypothesis from research log
    research_log_path = experiment_dir / "RESEARCH_LOG.md"
    if research_log_path.exists():
        research_log = research_log_path.read_text()
        hypothesis = _extract_hypothesis(research_log)
        if hypothesis:
            sections.append(f"## Hypothesis\n\n{hypothesis}\n")

        findings = _extract_findings(research_log)
        if findings:
            sections.append("## Key Findings\n")
            for finding in findings:
                sections.append(f"- {finding}")
            sections.append("")

    # Model card summary
    model_card_path = experiment_dir / "model_card.md"
    if model_card_path.exists():
        model_card = model_card_path.read_text()
        if "[" not in model_card.split("\n")[3]:  # Not template placeholder
            sections.append("## Model Details\n")
            sections.append(model_card)

    # Results summary from notes
    notes_path = experiment_dir / "notes.md"
    if notes_path.exists():
        notes = notes_path.read_text()
        results_match = re.search(r"## Results Summary\n(.+?)(?=\n##|\Z)", notes, re.DOTALL)
        if results_match:
            results = results_match.group(1).strip()
            if not results.startswith("["):
                sections.append(f"## Results\n\n{results}\n")

    # Paper reference
    paper_path = experiment_dir / "paper.md"
    if paper_path.exists():
        paper = paper_path.read_text()
        citation_match = re.search(r"## Citation\n(.+?)(?=\n##|\Z)", paper, re.DOTALL)
        if citation_match:
            citation = citation_match.group(1).strip()
            if citation:
                sections.append(f"## Reference\n\n{citation}\n")

    return "\n".join(sections)


def publish_to_wiki(experiment_dir: Path, wiki_dir: Path | None = None) -> Path | None:
    """Write experiment findings as a wiki page.

    Args:
        experiment_dir: Path to experiment directory
        wiki_dir: Path to llm-wiki repo (auto-detected if None)

    Returns:
        Path to created wiki page, or None if wiki not found
    """
    wiki_dir = wiki_dir or WIKI_DIR

    if not wiki_dir.exists():
        logger.warning("llm-wiki not found at %s — skipping wiki publish", wiki_dir)
        return None

    # Write to wiki's data directory
    wiki_pages_dir = wiki_dir / "data" / "pages"
    if not wiki_pages_dir.exists():
        wiki_pages_dir = wiki_dir / "pages"
    wiki_pages_dir.mkdir(parents=True, exist_ok=True)

    page_content = generate_wiki_page(experiment_dir)
    page_name = f"experiment-{experiment_dir.name}.md"
    page_path = wiki_pages_dir / page_name

    page_path.write_text(page_content)
    logger.info("Wiki page written: %s", page_path)
    return page_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Publish experiment findings to llm-wiki")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--wiki-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    page_path = publish_to_wiki(args.experiment_dir, args.wiki_dir)
    if page_path:
        print(f"\nWiki page created: {page_path}")
    else:
        print("\nllm-wiki not found — page content printed above")


if __name__ == "__main__":
    main()
