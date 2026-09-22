from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import re
from typing import Any, Callable, Iterable
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

TEAM_MAP = {
    "ARI":"ari", "ATL":"atl", "BAL":"bal", "BOS":"bos", "CHC":"chc", "CHW":"chw",
    "CIN":"cin", "CLE":"cle", "COL":"col", "DET":"det", "HOU":"hou", "KC":"kc",
    "KCR":"kc", "LAA":"laa", "LAD":"lad", "MIA":"mia", "MIL":"mil", "MIN":"min",
    "NYM":"nym", "NYY":"nyy", "OAK":"oak", "ATH":"oak", "PHI":"phi", "PIT":"pit",
    "SD":"sd", "SDP":"sd", "SEA":"sea", "SF":"sf", "SFG":"sf", "STL":"stl",
    "TB":"tb", "TBR":"tb", "TEX":"tex", "TOR":"tor", "WSH":"wsh", "WSN":"wsh",
}


@dataclass(frozen=True)
class MediaAsset:
    url: str | None
    source: str
    fallback_url: str | None = None


def _normalize_name(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return " ".join(text.split())


def team_logo_url(team_abbr: str) -> str | None:
    key = TEAM_MAP.get(str(team_abbr or "").upper().strip())
    return f"https://a.espncdn.com/i/teamlogos/mlb/500/{key}.png" if key else None


def _default_fetch_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": "DrewDFSCommandCenter/1.0"})
    with urlopen(req, timeout=4) as response:  # nosec B310 - fixed ESPN host is built by caller
        return json.loads(response.read().decode("utf-8"))


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _candidate_name(item: dict[str, Any]) -> str:
    for key in ("displayName", "name", "fullName", "shortName"):
        if item.get(key):
            return str(item[key])
    athlete = item.get("athlete")
    if isinstance(athlete, dict):
        return _candidate_name(athlete)
    return ""


def _candidate_id(item: dict[str, Any]) -> str | None:
    for key in ("id", "uid"):
        val = item.get(key)
        if val is not None:
            text = str(val)
            match = re.search(r"(?:athlete:)?(\d+)$", text)
            if match:
                return match.group(1)

    def from_link(value: Any) -> str | None:
        if isinstance(value, str):
            match = re.search(r"/id/(\d+)(?:/|$)", value)
            return match.group(1) if match else None
        if isinstance(value, dict):
            for child in value.values():
                found = from_link(child)
                if found:
                    return found
        if isinstance(value, list):
            for child in value:
                found = from_link(child)
                if found:
                    return found
        return None

    for key in ("link", "links"):
        found = from_link(item.get(key))
        if found:
            return found

    athlete = item.get("athlete")
    if isinstance(athlete, dict):
        return _candidate_id(athlete)
    return None


def _candidate_headshot(item: dict[str, Any]) -> str | None:
    for key in ("headshot", "image", "photo"):
        val = item.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
        if isinstance(val, dict):
            for href_key in ("href", "url", "src"):
                href = val.get(href_key)
                if isinstance(href, str) and href.startswith("http"):
                    return href
    athlete = item.get("athlete")
    if isinstance(athlete, dict):
        return _candidate_headshot(athlete)
    return None


def _resolve_from_payload(player_name: str, payload: Any, fallback_url: str | None) -> MediaAsset:
    target = _normalize_name(player_name)
    if not target:
        return MediaAsset(fallback_url, "team-logo-fallback" if fallback_url else "unavailable", fallback_url)
    candidates: list[tuple[int, dict[str, Any]]] = []
    for item in _walk_dicts(payload):
        name = _normalize_name(_candidate_name(item))
        if not name:
            continue
        score = 0
        if name == target:
            score = 100
        elif target in name or name in target:
            score = 60
        elif set(target.split()) == set(name.split()):
            score = 80
        if score:
            candidates.append((score, item))
    if not candidates:
        return MediaAsset(fallback_url, "team-logo-fallback" if fallback_url else "unavailable", fallback_url)
    _, best = max(candidates, key=lambda x: x[0])
    href = _candidate_headshot(best)
    if href:
        return MediaAsset(href, "espn-headshot", fallback_url)
    player_id = _candidate_id(best)
    if player_id:
        return MediaAsset(
            f"https://a.espncdn.com/i/headshots/mlb/players/full/{player_id}.png",
            "espn-id-headshot",
            fallback_url,
        )
    return MediaAsset(fallback_url, "team-logo-fallback" if fallback_url else "unavailable", fallback_url)


def _lookup_uncached(player_name: str, team_abbr: str | None, fetch_json: Callable[[str], Any]) -> MediaAsset:
    fallback = team_logo_url(team_abbr or "")
    try:
        url = f"https://site.web.api.espn.com/apis/search/v2?query={quote_plus(str(player_name))}&limit=10&sport=baseball"
        payload = fetch_json(url)
        return _resolve_from_payload(player_name, payload, fallback)
    except Exception:
        return MediaAsset(fallback, "team-logo-fallback" if fallback else "unavailable", fallback)


@lru_cache(maxsize=512)
def _cached_default_lookup(player_name: str, team_abbr: str | None) -> MediaAsset:
    return _lookup_uncached(player_name, team_abbr, _default_fetch_json)


def resolve_player_headshot(
    player_name: str,
    team_abbr: str | None = None,
    fetch_json: Callable[[str], Any] | None = None,
) -> MediaAsset:
    normalized_name = " ".join(str(player_name or "").split())
    normalized_team = str(team_abbr or "").upper().strip() or None
    if fetch_json is not None:
        return _lookup_uncached(normalized_name, normalized_team, fetch_json)
    return _cached_default_lookup(normalized_name, normalized_team)


def clear_media_cache() -> None:
    _cached_default_lookup.cache_clear()


def media_cache_info():
    return _cached_default_lookup.cache_info()
