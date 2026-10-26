"""Path resolution and write protection for the DART development workspace.

The collaborator's artifacts live under ``DART_Spike_v2_Clean/`` inside a shared
Google Drive folder. Anything written there syncs back and becomes visible to
them, so this module treats that tree as strictly read-only and routes every
write into ``dart_dev/``.

The project is developed both locally and in Colab, where the same Drive folder
is mounted at a different path. :func:`resolve_project_root` handles both.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "MASTER_ROOT",
    "DEV_ROOT",
    "MASTER",
    "DEV",
    "assert_writable",
    "resolve_project_root",
]


# ---------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------

#: Candidate locations of the ``ICAIF26`` folder, in priority order.
_ROOT_CANDIDATES = (
    # Colab, Drive mounted at /content/drive
    Path("/content/drive/MyDrive/ICAIF26"),
    # macOS Google Drive desktop client, shared-folder shortcut (the author's own path is not published: see the README, section Redactions)
    Path(
        "/Users/<user>/Library/CloudStorage/"
        "GoogleDrive-<account>/.shortcut-targets-by-id/"
        "<folder-id>/ICAIF26"
    ),
)


def resolve_project_root() -> Path:
    """Locate the ``ICAIF26`` folder for the current runtime.

    Resolution order:

    1. the ``DART_PROJECT_ROOT`` environment variable, when set;
    2. the first existing entry in :data:`_ROOT_CANDIDATES`;
    3. the parent of ``dart_dev``, inferred from this file's location.

    Raises:
        FileNotFoundError: if no candidate contains ``DART_Spike_v2_Clean``.
    """
    override = os.environ.get("DART_PROJECT_ROOT")
    if override:
        candidates: tuple[Path, ...] = (Path(override),)
    else:
        candidates = _ROOT_CANDIDATES + (Path(__file__).resolve().parents[2],)

    for candidate in candidates:
        if (candidate / "DART_Spike_v2_Clean").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate the ICAIF26 project root. Set DART_PROJECT_ROOT to "
        "the folder containing DART_Spike_v2_Clean/. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


PROJECT_ROOT: Path = resolve_project_root()

#: The collaborator's tree. Read-only.
MASTER_ROOT: Path = PROJECT_ROOT / "DART_Spike_v2_Clean"

#: This workspace. Writable.
DEV_ROOT: Path = PROJECT_ROOT / "dart_dev"

#: The redirected development project tree used by
#: ``DART_Spike_v2_Erce_Dev.ipynb``. Writable — authorized 2026-07-20 for CAISO
#: ingestion, which produces project data rather than library artifacts and so
#: does not belong under ``dart_dev/``.
#:
#: This is a *separate* authorization from ``DEV_ROOT``. The master tree remains
#: refused unconditionally.
DEV_PROJECT_ROOT: Path = PROJECT_ROOT / "DART_Spike_v2_Erce_Dev"


class _MasterPaths:
    """Read-only handles into the collaborator's cached artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root

        self.data = root / "data"
        self.panels = self.data / "panels"
        self.folds = self.data / "folds"
        self.standardized = self.data / "standardized"
        self.manifests = root / "manifests"
        self.results = root / "results"
        self.artifacts = root / "artifacts"

        # Certified inputs the development work reads from.
        self.price_panel = self.panels / "hourly_da_rt_price_panel.parquet"
        self.enriched_panel = (
            self.panels / "hourly_da_rt_price_panel_enriched.parquet"
        )
        self.feature_panel = (
            self.panels / "features" / "g1_g6_feature_panel_leakage_certified.parquet"
        )
        self.signed_tail_labels = (
            self.folds / "signed_tail_labels" / "signed_tail_development_labels.parquet"
        )
        self.signed_tail_thresholds = (
            self.folds / "signed_tail_labels" / "signed_tail_thresholds.csv"
        )
        self.fold_registry = (
            self.folds / "walk_forward" / "walk_forward_fold_registry.csv"
        )
        self.fold_membership = (
            self.folds / "walk_forward" / "walk_forward_row_membership.parquet"
        )
        self.holdout_membership = (
            self.folds / "walk_forward" / "final_holdout_row_membership.parquet"
        )
        self.information_policy = self.manifests / "feature_information_policy.json"
        self.feature_definitions = self.manifests / "feature_definition_registry.csv"
        self.feature_groups = self.manifests / "feature_group_registry.csv"
        self.scope_freeze = self.manifests / "experiment_scope_freeze_manifest.json"


class _DevPaths:
    """Writable locations for development work."""

    def __init__(self, root: Path) -> None:
        self.root = root

        self.src = root / "src"
        self.tests = root / "tests"

        #: Derived tables. Disposable — deleting this costs time, not information.
        self.cache = root / "cache"

        #: Experiment results worth keeping.
        self.outputs = root / "outputs"
        self.tables = self.outputs / "tables"
        self.figures = self.outputs / "figures"
        self.manifests = self.outputs / "manifests"

    def ensure(self) -> None:
        """Create the writable directories if they do not yet exist."""
        for path in (
            self.cache,
            self.outputs,
            self.tables,
            self.figures,
            self.manifests,
        ):
            path.mkdir(parents=True, exist_ok=True)


MASTER = _MasterPaths(MASTER_ROOT)
DEV = _DevPaths(DEV_ROOT)


# ---------------------------------------------------------------------
# Write protection
# ---------------------------------------------------------------------

def assert_writable(path: str | Path, allow_project_tree: bool = False) -> Path:
    """Confirm that ``path`` is safe to write to, and return it resolved.

    A path is writable when it lies inside ``dart_dev/``. Everything under the
    collaborator's ``DART_Spike_v2_Clean/`` tree is rejected unconditionally, as
    is anything outside the project root.

    Args:
        path: The intended write target.
        allow_project_tree: Additionally permit ``DART_Spike_v2_Erce_Dev/``.
            Off by default so that ordinary library code cannot reach the
            project tree by accident — ingestion scripts opt in explicitly.

    Raises:
        PermissionError: if the path would write into shared or foreign space.
    """
    resolved = Path(path).expanduser().resolve()

    # The master tree is refused first and always. No flag unlocks it.
    if resolved.is_relative_to(MASTER_ROOT.resolve()):
        raise PermissionError(
            f"Refusing to write to the collaborator's tree: {resolved}\n"
            "Everything under DART_Spike_v2_Clean/ is read-only. Write to "
            f"{DEV.outputs} or {DEV.cache} instead."
        )

    permitted = [DEV_ROOT.resolve()]
    if allow_project_tree:
        permitted.append(DEV_PROJECT_ROOT.resolve())

    if not any(resolved.is_relative_to(root) for root in permitted):
        allowed = " or ".join(str(root) for root in permitted)
        raise PermissionError(
            f"Refusing to write outside the permitted roots: {resolved}\n"
            f"Permitted: {allowed}\n"
            "Ask before writing anywhere else in the project."
        )

    return resolved
