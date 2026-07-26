"""Chart selection — a rules + scoring hybrid.

Two layers, deliberately separated:

**Rules** (``rules.yaml``) map a data *shape* to candidate chart types. They are broad and
generous: a datetime plus a measure proposes line, area, and scatter, and does not try to decide
between them.

**Scoring** ranks those candidates against the actual data — cardinality, missingness,
distribution skew, correlation strength, and readability limits. This is where "you have 400
categories, a bar chart will be unreadable" gets expressed, as a penalty rather than a
prohibition.

Every returned :class:`~plotaviz.core.spec.ChartSpec` carries a ``why`` string assembled from the
rule's rationale plus whatever the scoring layer had to say. The user always sees the reasoning,
and the ranked alternatives are all real specs they can switch to.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from .errors import SelectionError
from .profiler import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, TEXT, DatasetProfile, as_float
from .spec import MATRIX_CHARTS, ChartSpec

#: Location of the shipped rules file.
RULES_PATH = Path(__file__).with_name("rules.yaml")

#: How many columns per role a rule slot is allowed to explore. Keeps the candidate space small.
_MAX_COLUMNS_PER_SLOT = 3

#: Cap on returned recommendations.
DEFAULT_TOP_K = 8

_ROLE_TOKENS = {
    "numeric": NUMERIC,
    "categorical": CATEGORICAL,
    "datetime": DATETIME,
    "boolean": BOOLEAN,
    "text": TEXT,
}


@dataclass
class RulesConfig:
    """The parsed contents of ``rules.yaml``.

    Attributes:
        rules: Rule definitions in file order.
        scoring: Scoring caps and weights.
        version: Config schema version.
    """

    rules: list[dict[str, Any]]
    scoring: dict[str, Any]
    version: int = 1

    @property
    def weights(self) -> dict[str, float]:
        """Scoring weights, with sane defaults if the file omits any."""
        defaults = {
            "cardinality_penalty": 0.35,
            "missing_penalty": 0.25,
            "correlation_bonus": 0.30,
            "skew_bonus": 0.15,
            "timespan_bonus": 0.10,
            "identifier_penalty": 0.60,
            "constant_penalty": 0.50,
        }
        return {**defaults, **(self.scoring.get("weights") or {})}

    def cap(self, name: str, default: int) -> int:
        """Read a readability cap such as ``max_bar_categories``."""
        try:
            return int(self.scoring.get(name, default))
        except (TypeError, ValueError):
            return default


def load_rules(path: str | Path | None = None) -> RulesConfig:
    """Load the selection rules.

    Args:
        path: Alternate rules file. Defaults to the one shipped with the package, so users can
            point at their own without editing the install.

    Raises:
        SelectionError: If the file is missing or is not valid YAML with a ``rules`` list.
    """
    target = Path(path) if path else RULES_PATH
    try:
        with open(target, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise SelectionError(
            f"Chart rules file not found at {target}.",
            hint="Reinstall PlotaViz, or point at your own rules file.",
        ) from exc
    except yaml.YAMLError as exc:
        raise SelectionError(
            f"Chart rules file {target.name} is not valid YAML.", hint=str(exc)
        ) from exc

    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        raise SelectionError(f"{target.name} contains no rules.")

    return RulesConfig(
        rules=rules, scoring=data.get("scoring") or {}, version=int(data.get("version", 1))
    )


class ChartSelector:
    """Turns a :class:`~plotaviz.core.profiler.DatasetProfile` into ranked chart specs.

    Args:
        config: Loaded rules. Defaults to the shipped ``rules.yaml``.
    """

    #: At most this many variants of one chart type survive into the ranked alternatives. Without
    #: a cap, a wide dataset returns eight line charts and the user has no real choice to make.
    max_per_chart_type: int = 2

    def __init__(self, config: RulesConfig | None = None) -> None:
        self.config = config or load_rules()

    # ------------------------------------------------------------------ public API

    def recommend(
        self,
        profile: DatasetProfile,
        *,
        top_k: int = DEFAULT_TOP_K,
        prefer_columns: list[str] | None = None,
    ) -> list[ChartSpec]:
        """Rank chart specs for a dataset, best first.

        Args:
            profile: The dataset profile.
            top_k: Maximum number of recommendations to return.
            prefer_columns: Columns the user has signalled interest in (selected in the UI, or
                named in a natural-language query). Candidates using them are boosted.

        Returns:
            Ranked :class:`ChartSpec` objects, each with a populated ``why`` and ``score``.

        Raises:
            SelectionError: If no rule matches — a dataset of nothing but identifiers, say.
        """
        columns_by_role = self._rank_columns(profile)
        candidates = self._candidates_for(profile, columns_by_role)

        if not candidates and any(prof.is_identifier for prof in profile.columns.values()):
            # Every usable column was filtered out as an identifier. Rather than refusing to draw
            # anything — which is what a table of IDs and a dense integer key looks like — offer
            # the identifier columns too, and say so in the justification. A chart the user can
            # correct beats a dead end.
            columns_by_role = self._rank_columns(profile, include_identifiers=True)
            candidates = self._candidates_for(profile, columns_by_role)
            for spec in candidates:
                spec.why += (
                    " Every column in this dataset looks like an identifier, so this chart uses "
                    "one anyway — use the type override panel if that is wrong."
                )

        if not candidates:
            raise SelectionError(
                "PlotaViz could not find a chart that suits this dataset.",
                hint=(
                    "This usually means every column looks like an identifier or free text. "
                    "Use the type override panel to mark a column as numeric, categorical, or a "
                    "date."
                ),
            )

        # Rank on the unclipped score. `ChartSpec.score` is a 0–1 confidence for display, and
        # several strong candidates can all saturate at 1.0 — sorting on that would scramble a
        # ranking the raw numbers get right.
        scored = [self._score(spec, profile, prefer_columns or []) for spec in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return self._dedupe([spec for spec, _ in scored])[:top_k]

    def best(self, profile: DatasetProfile, **kwargs: Any) -> ChartSpec:
        """The single top recommendation. Convenience for CLI ``--auto`` mode."""
        return self.recommend(profile, **kwargs)[0]

    def is_ambiguous(self, ranked: list[ChartSpec], *, margin: float = 0.06) -> bool:
        """Whether the top two recommendations are close enough to be a coin flip.

        The UI uses this to decide when consulting an LLM is worth the latency: a clear winner
        needs no second opinion.
        """
        return len(ranked) >= 2 and (ranked[0].score - ranked[1].score) < margin

    # ------------------------------------------------------------------ internals

    def _candidates_for(
        self, profile: DatasetProfile, columns: dict[str, list[str]]
    ) -> list[ChartSpec]:
        """Expand every rule that matches the available columns into concrete specs."""
        candidates: list[ChartSpec] = []
        for rule in self.config.rules:
            if not self._rule_matches(rule, columns):
                continue
            for candidate in rule.get("candidates", []):
                candidates.extend(self._expand(rule, candidate, columns, profile))
        return candidates

    def _rank_columns(
        self, profile: DatasetProfile, *, include_identifiers: bool = False
    ) -> dict[str, list[str]]:
        """Order each role's columns by how good an axis they would make.

        Identifiers are excluded unless the caller asks for them back. Within a role, prefer
        complete columns, then moderate cardinality (a 4-category column beats a 400-category one
        for an axis).
        """

        def sort_key(name: str, *, role: str) -> tuple[float, float]:
            prof = profile.columns[name]
            if role in (CATEGORICAL, TEXT):
                # Distance from an ideal ~6 categories, so tiny and huge both rank lower.
                cardinality_cost = abs(prof.n_unique - 6) / 100.0
            else:
                cardinality_cost = 0.0
            return (prof.pct_missing / 100.0, cardinality_cost)

        ranked: dict[str, list[str]] = {}
        for role in _ROLE_TOKENS.values():
            names = profile.by_role(role, include_identifiers=include_identifiers)
            ranked[role] = sorted(names, key=partial(sort_key, role=role))
        return ranked

    def _rule_matches(self, rule: dict[str, Any], columns: dict[str, list[str]]) -> bool:
        """Whether the dataset has at least the columns a rule requires."""
        requires = rule.get("requires") or {}
        for token, needed in requires.items():
            role = _ROLE_TOKENS.get(str(token))
            if role is None:
                return False
            if len(columns.get(role, [])) < int(needed):
                return False
        return True

    def _expand(
        self,
        rule: dict[str, Any],
        candidate: dict[str, Any],
        columns: dict[str, list[str]],
        profile: DatasetProfile,
    ) -> list[ChartSpec]:
        """Turn one rule candidate into concrete specs by resolving column placeholders.

        A candidate like ``{x: categorical.0, y: numeric.0}`` produces several specs — the top
        few columns of each role — so the user gets real alternatives rather than one guess.
        """
        mapping = candidate.get("map") or {}
        if not mapping and candidate.get("chart") in MATRIX_CHARTS:
            return [
                ChartSpec(
                    chart=str(candidate["chart"]),
                    why=str(candidate.get("why", "")),
                    score=float(candidate.get("base", 0.5)),
                    options=dict(candidate.get("options") or {}),
                    agg=candidate.get("agg"),
                )
            ]

        # Resolve every placeholder to the list of columns it could mean.
        options_per_slot: dict[str, list[str | None]] = {}
        for slot, token in mapping.items():
            token = str(token)
            optional = token.startswith("?")
            role_name, _, index_text = token.lstrip("?").partition(".")
            role = _ROLE_TOKENS.get(role_name)
            if role is None:
                return []
            available = columns.get(role, [])
            try:
                start = int(index_text) if index_text else 0
            except ValueError:
                start = 0

            picks: list[str | None] = list(available[start : start + _MAX_COLUMNS_PER_SLOT])
            if optional:
                picks = [None, *picks[:1]]  # no grouping, or the single best grouping column
            if not picks:
                if not optional:
                    return []
                picks = [None]
            options_per_slot[slot] = picks

        specs: list[ChartSpec] = []
        slots = list(options_per_slot)
        for combo in itertools.product(*(options_per_slot[s] for s in slots)):
            assignment = dict(zip(slots, combo, strict=True))
            chosen = [c for c in combo if c is not None]
            if len(set(chosen)) != len(chosen):
                continue  # the same column mapped twice is never meaningful

            spec = ChartSpec(
                chart=str(candidate["chart"]),
                x=assignment.get("x"),
                y=assignment.get("y"),
                color=assignment.get("color"),
                agg=candidate.get("agg"),
                options=dict(candidate.get("options") or {}),
                why=str(candidate.get("why", "")),
                score=float(candidate.get("base", 0.5)),
                source="rules",
            )
            spec.options.setdefault("rule", rule.get("name", ""))
            try:
                spec.validate(list(profile.columns))
            except Exception:
                continue
            specs.append(spec)

        return specs

    def _score(
        self, spec: ChartSpec, profile: DatasetProfile, prefer: list[str]
    ) -> tuple[ChartSpec, float]:
        """Adjust a candidate's base score against the real data, recording the reasons.

        Returns:
            ``(spec, raw_score)``. The spec carries the clipped 0–1 confidence for display; the
            raw score is unbounded and is what the ranking sorts on.
        """
        weights = self.config.weights
        score = spec.score
        notes: list[str] = []

        # --- readability: too many categories on a categorical axis
        cap_name = {
            "pie": "max_pie_categories",
            "treemap": "max_treemap_categories",
        }.get(spec.chart, "max_bar_categories")
        cap = self.config.cap(cap_name, 30)

        axis_col = spec.x if spec.chart not in MATRIX_CHARTS else None
        if axis_col and axis_col in profile.columns:
            prof = profile.columns[axis_col]
            if prof.role in (CATEGORICAL, TEXT, BOOLEAN):
                if prof.n_unique > cap:
                    overflow = min(1.0, (prof.n_unique - cap) / max(cap, 1))
                    score -= weights["cardinality_penalty"] * overflow
                    notes.append(
                        f"{axis_col!r} has {prof.n_unique:,} categories, well past the {cap} "
                        "that stay readable, so this ranks lower."
                    )
                elif spec.chart == "treemap" and prof.n_unique > 12:
                    score += 0.12
                    notes.append(
                        f"{prof.n_unique} categories is more than a bar chart handles "
                        "comfortably, which suits a treemap."
                    )
            if prof.n_unique <= 1:
                score -= weights["constant_penalty"]
                notes.append(f"{axis_col!r} has a single value, so there is nothing to compare.")

        # --- grouping series count
        if spec.color and spec.color in profile.columns:
            n_series = profile.columns[spec.color].n_unique
            max_series = self.config.cap("max_grouped_series", 8)
            if n_series > max_series:
                score -= weights["cardinality_penalty"] * min(1.0, n_series / (max_series * 4))
                notes.append(
                    f"Splitting by {spec.color!r} would draw {n_series:,} series, which is too "
                    "many to distinguish."
                )
            elif 2 <= n_series <= max_series:
                score += 0.05
                notes.append(f"Splitting by {spec.color!r} adds {n_series} comparable series.")

        # --- missing data across the mapped columns
        used = [c for c in spec.columns_used if c in profile.columns]
        if used:
            worst = max(profile.columns[c].pct_missing for c in used)
            if worst > 5:
                score -= weights["missing_penalty"] * min(1.0, worst / 100.0)
                notes.append(f"Up to {worst:.0f}% of values in the columns used are missing.")

        # --- identifiers should almost never be plotted
        if any(profile.columns[c].is_identifier for c in used):
            score -= weights["identifier_penalty"]
            notes.append(
                "One of the mapped columns looks like an identifier rather than a measure."
            )

        # --- correlation strength for scatter
        if spec.chart == "scatter" and spec.x and spec.y and profile.correlations is not None:
            corr = profile.correlations
            if spec.x in corr.columns and spec.y in corr.columns:
                r = as_float(corr.loc[spec.x, spec.y])
                if r is not None:
                    score += weights["correlation_bonus"] * abs(r)
                    strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
                    direction = "positive" if r > 0 else "negative"
                    notes.append(
                        f"These two columns have a {strength} {direction} correlation "
                        f"(r = {r:.2f})."
                    )
                    if abs(r) > 0.4:
                        spec.options.setdefault("trendline", True)

        # --- skew favours distribution-shape charts and log scales
        if spec.y and spec.y in profile.columns:
            skew = abs(profile.columns[spec.y].stats.get("skew", 0.0))
            if skew > 2:
                if spec.chart in {"box", "violin"}:
                    score += weights["skew_bonus"]
                    notes.append(
                        f"{spec.y!r} is heavily skewed, which this chart shows better than "
                        "an average."
                    )
                elif spec.chart in {"bar", "line", "scatter"}:
                    spec.options.setdefault("log_y", True)
                    notes.append(f"{spec.y!r} is heavily skewed, so a log y axis is suggested.")

        # --- a real time span makes a time series more compelling
        if spec.chart in {"line", "area"} and spec.x and spec.x in profile.columns:
            prof = profile.columns[spec.x]
            if prof.role == DATETIME:
                periods = prof.stats.get("n_periods", 0)
                if periods >= 10:
                    score += weights["timespan_bonus"]
                    notes.append(
                        f"{int(periods):,} distinct time points give the trend real shape."
                    )
                elif periods < 3:
                    score -= 0.25
                    notes.append(
                        f"Only {int(periods)} distinct time point(s) — too few to show a trend."
                    )

        # --- histogram bin sanity
        if spec.chart in {"histogram", "kde"} and profile.n_rows < self.config.cap(
            "ideal_hist_rows", 30
        ):
            score -= 0.15
            notes.append(f"Only {profile.n_rows} rows, which makes the distribution shape noisy.")

        # --- user or NL query interest
        if prefer:
            hits = [c for c in used if c in prefer]
            if hits:
                score += 0.15 * len(hits)
                notes.append(f"Uses the column(s) you asked about: {', '.join(hits)}.")

        spec.score = max(0.0, min(1.0, score))
        spec.why = " ".join([spec.why, *notes]).strip()
        return spec, score

    def _dedupe(self, specs: list[ChartSpec]) -> list[ChartSpec]:
        """Drop duplicate mappings and keep the alternatives genuinely varied.

        Two passes: exact duplicate mappings go, then each chart type is capped so the list the
        user picks from offers different *kinds* of chart rather than eight near-identical ones.
        Anything squeezed out by the cap is appended after the varied set, so nothing is lost.
        """
        seen: set[tuple[Any, ...]] = set()
        unique: list[ChartSpec] = []
        for spec in specs:
            key = (spec.chart, spec.x, spec.y, spec.color, spec.agg)
            if key in seen:
                continue
            seen.add(key)
            unique.append(spec)

        per_type: dict[str, int] = {}
        varied: list[ChartSpec] = []
        overflow: list[ChartSpec] = []
        for spec in unique:
            count = per_type.get(spec.chart, 0)
            if count < self.max_per_chart_type:
                per_type[spec.chart] = count + 1
                varied.append(spec)
            else:
                overflow.append(spec)
        return varied + overflow


def recommend(
    profile: DatasetProfile, *, top_k: int = DEFAULT_TOP_K, rules_path: str | Path | None = None
) -> list[ChartSpec]:
    """Convenience wrapper: build a selector and rank charts in one call."""
    return ChartSelector(load_rules(rules_path)).recommend(profile, top_k=top_k)
