"""GitHub API client — fetch repository README and file tree.

Uses the ``ghapi`` library for authenticated access to the GitHub API.
The token is read from the ``GITHUB_TOKEN`` environment variable.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field

import requests
from ghapi.all import GhApi


def _get_token() -> str | None:
    """Read GitHub token from environment."""
    return os.environ.get("GITHUB_TOKEN")


def _parse_github_url(url: str) -> tuple[str, str, str | None]:
    """Extract ``(owner, repo, sub_path)`` from a GitHub URL.

    *sub_path* is set when the URL points to a specific file
    (e.g. ``/blob/main/docs/README.md``), otherwise ``None``.
    """
    url = re.sub(r"\.git$", "", url)
    cleaned = re.sub(r"https?://github\.com/", "", url).strip("/")
    parts = cleaned.split("/")

    owner = parts[0] if len(parts) > 0 else ""
    repo = parts[1] if len(parts) > 1 else ""

    # Detect sub-path after /blob/<branch>/ or /tree/<branch>/
    sub_path: str | None = None
    if len(parts) > 3 and parts[2] in ("blob", "tree"):
        # parts[3] is the branch name, everything after is the path
        sub_path = "/".join(parts[4:]) if len(parts) > 4 else None

    return owner, repo, sub_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class GitHubContent:
    """Parsed content from a GitHub repository."""

    readme: str = ""
    files: list[str] = field(default_factory=list)


def fetch_github_content(github_url: str) -> GitHubContent:
    """Fetch README and file tree from a GitHub repository.

    Parameters
    ----------
    github_url : str
        Any GitHub URL — can be a repo root or a direct file link.

    Returns
    -------
    GitHubContent
        Contains the README text and a flat list of file paths.
    """
    owner, repo, sub_path = _parse_github_url(github_url)
    if not owner or not repo:
        return GitHubContent()

    token = _get_token()
    api = GhApi(owner=owner, repo=repo, token=token)

    readme = _fetch_readme(api, github_url, sub_path)
    files = _fetch_file_tree(api)

    return GitHubContent(readme=readme, files=files)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _fetch_readme(api: GhApi, original_url: str, sub_path: str | None) -> str:
    """Fetch the README content from the repository.

    If the original URL points directly to a ``.md`` file, fetch that file
    via raw.githubusercontent.com.  Otherwise, use the GitHub API to locate
    the repo's default README.
    """
    # Case 1: URL points to a specific .md file
    if original_url.lower().endswith(".md") and sub_path:
        raw_path = re.sub(r"/(?:blob|tree)/", "/refs/heads/", original_url)
        raw_path = re.sub(r"https?://github\.com/", "", raw_path)
        try:
            resp = requests.get(f"https://raw.githubusercontent.com/{raw_path}", timeout=15)
            if resp.status_code == 200:
                return resp.text
        except requests.RequestException:
            pass

    # Case 2: Use GitHub API to get default README
    try:
        readme_info = api.repos.get_readme()
        download_url = readme_info.get("download_url", "")
        if download_url:
            resp = requests.get(download_url, timeout=15)
            if resp.status_code == 200:
                return resp.text

        # Fallback: decode base64 content from the API response
        content_b64 = readme_info.get("content", "")
        if content_b64:
            return base64.b64decode(content_b64).decode("utf-8")
    except Exception:
        pass

    return ""


def _fetch_file_tree(api: GhApi) -> list[str]:
    """Fetch the complete file tree of the repository's default branch."""
    files: list[str] = []

    try:
        # Try main, then master
        for branch in ("main", "master"):
            try:
                ref = api.git.get_ref(f"heads/{branch}")
                break
            except Exception:
                continue
        else:
            return files

        commit_sha = ref.object.sha
        commit = api.git.get_commit(commit_sha)
        tree_sha = commit["tree"]["sha"]
        tree = api.git.get_tree(tree_sha, recursive=True)

        for item in tree.tree:
            if item.type == "blob":
                files.append(item.path)

    except Exception:
        pass

    return files
