from __future__ import annotations

from functools import lru_cache
import json
import math
import re
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

TEAM_ALIASES = {
    # Research-sheet abbreviations and common MLB abbreviations.
    "KC":"KCR", "KCR":"KCR", "SD":"SDP", "SDP":"SDP", "SF":"SFG", "SFG":"SFG",
    "TB":"TBR", "TBR":"TBR", "WSH":"WAS", "WSN":"WAS", "WAS":"WAS",
    "CWS":"CHW", "CHW":"CHW", "OAK":"ATH", "ATH":"ATH",
    "ARI":"ARI", "ATL":"ATL", "BAL":"BAL", "BOS":"BOS", "CHC":"CHC", "CIN":"CIN",
    "CLE":"CLE", "COL":"COL", "DET":"DET", "HOU":"HOU", "LAA":"LAA", "LAD":"LAD",
    "MIA":"MIA", "MIL":"MIL", "MIN":"MIN", "NYM":"NYM", "NYY":"NYY", "PHI":"PHI",
    "PIT":"PIT", "SEA":"SEA", "STL":"STL", "TEX":"TEX", "TOR":"TOR",
    # Full MLB team names are included because schedule payloads are not guaranteed
    # to hydrate the abbreviation field in every environment/version.
    "ARIZONA DIAMONDBACKS":"ARI", "ATLANTA BRAVES":"ATL", "BALTIMORE ORIOLES":"BAL",
    "BOSTON RED SOX":"BOS", "CHICAGO CUBS":"CHC", "CHICAGO WHITE SOX":"CHW",
    "CINCINNATI REDS":"CIN", "CLEVELAND GUARDIANS":"CLE", "COLORADO ROCKIES":"COL",
    "DETROIT TIGERS":"DET", "HOUSTON ASTROS":"HOU", "KANSAS CITY ROYALS":"KCR",
    "LOS ANGELES ANGELS":"LAA", "LOS ANGELES DODGERS":"LAD", "MIAMI MARLINS":"MIA",
    "MILWAUKEE BREWERS":"MIL", "MINNESOTA TWINS":"MIN", "NEW YORK METS":"NYM",
    "NEW YORK YANKEES":"NYY", "ATHLETICS":"ATH", "OAKLAND ATHLETICS":"ATH",
    "PHILADELPHIA PHILLIES":"PHI", "PITTSBURGH PIRATES":"PIT", "SAN DIEGO PADRES":"SDP",
    "SAN FRANCISCO GIANTS":"SFG", "SEATTLE MARINERS":"SEA", "ST. LOUIS CARDINALS":"STL",
    "ST LOUIS CARDINALS":"STL", "TAMPA BAY RAYS":"TBR", "TEXAS RANGERS":"TEX",
    "TORONTO BLUE JAYS":"TOR", "WASHINGTON NATIONALS":"WAS",
}

# Multiplicative boost fields from the user's Advanced Ballpark sheet. Lower
# strikeout rate is favorable for hitters, so that factor is inverted.
CONTACT_SPECS = {
    "Hits_boost": (0.15, False),
    "Doubles_boost": (0.10, False),
    "K_rate_boost": (0.10, True),
    "Walk_rate_boost": (0.05, False),
    "xBA_avg_boost": (0.20, False),
    "xwOBA_avg_boost": (0.20, False),
    "BABIP_avg_boost": (0.10, False),
    "AVG_avg_boost": (0.10, False),
}
POWER_SPECS = {
    "Homeruns_boost": (0.40, False),
    "xSLG_avg_boost": (0.30, False),
    "xwOBA_avg_boost": (0.20, False),
    "Doubles_boost": (0.10, False),
}



def normalize_team(value: object) -> str:
    key = str(value or '').upper().strip()
    return TEAM_ALIASES.get(key, key)


def _weighted_ratio(row: pd.Series, specs: dict[str, tuple[float, bool]]) -> float:
    pairs = []
    for col, (weight, invert) in specs.items():
        value = pd.to_numeric(pd.Series([row.get(col)]), errors='coerce').iloc[0]
        if pd.notna(value) and value > 0:
            ratio = (1.0 / float(value)) if invert else float(value)
            pairs.append((ratio, weight))
    if not pairs:
        return float('nan')
    denom = sum(w for _, w in pairs)
    # Weighted geometric mean is appropriate because source columns are
    # multiplicative park factors centered around 1.0.
    return math.exp(sum((w / denom) * math.log(v) for v, w in pairs))


def _ratio_score(ratio: float) -> float:
    if pd.isna(ratio):
        return float('nan')
    # 20% favorable ~= 100, 20% suppressive ~= 0; bounded for stability.
    return float(np.clip(50.0 + (ratio - 1.0) * 250.0, 0.0, 100.0))


def build_ballpark_profiles(ballpark_df: pd.DataFrame | None) -> pd.DataFrame:
    if ballpark_df is None or ballpark_df.empty:
        return pd.DataFrame(columns=['Stadium','Split','Contact Environment','Power Environment','Ballpark Score','Confidence'])
    df = ballpark_df.copy()
    lookup = {re.sub(r'[^a-z0-9]', '', str(c).lower()): c for c in df.columns}
    stadium_col = lookup.get('stadium') or lookup.get('team')
    split_col = lookup.get('split') or lookup.get('bats')
    if stadium_col is None or split_col is None:
        raise ValueError('Ballpark Research requires Stadium and Split columns.')
    df['Stadium'] = df[stadium_col].map(normalize_team)
    df['Split'] = df[split_col].astype(str).str.upper().str.strip()
    rows = []
    for _, row in df.iterrows():
        contact_ratio = _weighted_ratio(row, CONTACT_SPECS)
        power_ratio = _weighted_ratio(row, POWER_SPECS)
        contact = _ratio_score(contact_ratio)
        power = _ratio_score(power_ratio)
        vals = [v for v in (contact, power) if pd.notna(v)]
        score = float(np.mean(vals)) if vals else np.nan
        pa = pd.to_numeric(pd.Series([row.get('PA')]), errors='coerce').iloc[0]
        confidence = float(np.clip((float(pa) / 5000.0) if pd.notna(pa) else .65, 0.25, 1.0))
        rows.append({
            'Stadium': row['Stadium'], 'Split': row['Split'], 'PA': pa,
            'Contact Ratio': contact_ratio, 'Power Ratio': power_ratio,
            'Contact Environment': contact, 'Power Environment': power,
            'Ballpark Score': score, 'Confidence': confidence,
        })
    return pd.DataFrame(rows)


def resolve_batter_side(bats: object, pitcher_throws: object) -> str | None:
    b = str(bats or '').upper().strip()
    p = str(pitcher_throws or '').upper().strip()
    if b == 'L': return 'LHH'
    if b == 'R': return 'RHH'
    if b == 'S':
        if p == 'R': return 'LHH'
        if p == 'L': return 'RHH'
    return None


def resolve_player_handedness_from_payload(payload: Any) -> dict[str, str | None]:
    people = payload.get('people', []) if isinstance(payload, dict) else []
    person = people[0] if people else {}
    bat = person.get('batSide', {}) if isinstance(person, dict) else {}
    pitch = person.get('pitchHand', {}) if isinstance(person, dict) else {}
    return {'bats': bat.get('code'), 'throws': pitch.get('code')}


def _fetch_json(url: str) -> Any:
    req = Request(url, headers={'User-Agent':'DrewDFSCommandCenter/1.0'})
    with urlopen(req, timeout=4) as response:  # nosec B310 - public MLB/Open-Meteo endpoints
        return json.loads(response.read().decode('utf-8'))


@lru_cache(maxsize=1024)
def lookup_player_handedness(player_name: str) -> dict[str, str | None]:
    name = ' '.join(str(player_name or '').split())
    if not name:
        return {'bats': None, 'throws': None}
    try:
        payload = _fetch_json(f'https://statsapi.mlb.com/api/v1/people/search?names={quote(name)}')
        return resolve_player_handedness_from_payload(payload)
    except Exception:
        return {'bats': None, 'throws': None}


def parse_schedule_payload(payload: Any) -> pd.DataFrame:
    rows = []
    for date_block in (payload.get('dates', []) if isinstance(payload, dict) else []):
        for game in date_block.get('games', []):
            teams = game.get('teams', {})
            away = teams.get('away', {}).get('team', {})
            home = teams.get('home', {}).get('team', {})
            venue = game.get('venue', {})
            rows.append({
                'Game PK': game.get('gamePk'),
                'Game Time': game.get('gameDate'),
                'Away Team': normalize_team(away.get('abbreviation') or away.get('name')),
                'Home Team': normalize_team(home.get('abbreviation') or home.get('name')),
                'Venue': venue.get('name'),
                'Venue ID': venue.get('id'),
                'Venue Team': normalize_team(home.get('abbreviation') or home.get('name')),
                'Away Probable Pitcher': teams.get('away', {}).get('probablePitcher', {}).get('fullName'),
                'Home Probable Pitcher': teams.get('home', {}).get('probablePitcher', {}).get('fullName'),
            })
    return pd.DataFrame(rows)


def fetch_mlb_schedule(slate_date: str, fetch_json: Callable[[str], Any] = _fetch_json) -> pd.DataFrame:
    try:
        url = ('https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=' + quote(str(slate_date)) +
               '&hydrate=team,venue,probablePitcher')
        return parse_schedule_payload(fetch_json(url))
    except Exception:
        return pd.DataFrame(columns=['Game PK','Game Time','Away Team','Home Team','Venue','Venue ID','Venue Team','Away Probable Pitcher','Home Probable Pitcher'])


def _neutral_blend(profiles: pd.DataFrame, stadium: str, column: str) -> float:
    rows = profiles[profiles['Stadium'] == normalize_team(stadium)]
    vals = pd.to_numeric(rows[column], errors='coerce').dropna()
    return float(vals.mean()) if not vals.empty else np.nan


def park_value(profiles: pd.DataFrame, stadium: str, split: str | None, column: str) -> float:
    if profiles is None or profiles.empty:
        return np.nan
    rows = profiles[profiles['Stadium'] == normalize_team(stadium)]
    if split:
        exact = rows[rows['Split'] == split]
        vals = pd.to_numeric(exact[column], errors='coerce').dropna() if column in exact else pd.Series(dtype=float)
        if not vals.empty:
            return float(vals.iloc[0])
    return _neutral_blend(profiles, stadium, column)


def build_player_environment(
    player_roo_df: pd.DataFrame | None,
    matchups: pd.DataFrame | None,
    profiles: pd.DataFrame | None,
    handedness: dict[str, dict[str, str | None]] | None = None,
) -> pd.DataFrame:
    if player_roo_df is None or player_roo_df.empty or matchups is None or matchups.empty or profiles is None or profiles.empty:
        return pd.DataFrame()
    handedness = handedness or {}
    df = player_roo_df.copy()
    def col(*names):
        norms={re.sub(r'[^a-z0-9]','',str(c).lower()):c for c in df.columns}
        for n in names:
            hit=norms.get(re.sub(r'[^a-z0-9]','',n.lower()))
            if hit is not None:return hit
        return None
    pcol, tcol, ocol, poscol = col('Player','Name'), col('Team'), col('Opp','Opponent'), col('Position','Pos')
    if not pcol or not tcol:
        return pd.DataFrame()
    matchup_by_team = {}
    for _, m in matchups.iterrows():
        away, home = normalize_team(m.get('Away Team')), normalize_team(m.get('Home Team'))
        matchup_by_team[away] = m
        matchup_by_team[home] = m
    rows=[]
    for _, r in df.iterrows():
        name=str(r.get(pcol,'')); team=normalize_team(r.get(tcol)); m=matchup_by_team.get(team)
        if m is None: continue
        home=normalize_team(m.get('Home Team')); away=normalize_team(m.get('Away Team'))
        is_pitcher = str(r.get(poscol,'')).upper().strip() in {'P','SP','SP1','SP2'} if poscol else False
        h = handedness.get(name) or lookup_player_handedness(name)
        bats, throws = h.get('bats'), h.get('throws')
        opposing_pitcher_name = m.get('Home Probable Pitcher') if team == away else m.get('Away Probable Pitcher')
        opp_hand = (handedness.get(str(opposing_pitcher_name), {}) or lookup_player_handedness(str(opposing_pitcher_name))).get('throws') if opposing_pitcher_name else None
        split = None if is_pitcher else resolve_batter_side(bats, opp_hand)
        contact = park_value(profiles, home, split, 'Contact Environment')
        power = park_value(profiles, home, split, 'Power Environment')
        base_score = np.nanmean([contact,power]) if any(pd.notna(x) for x in (contact,power)) else np.nan
        weather_score = m.get('Weather Score', np.nan)
        combined_score = _combine_park_weather(base_score, weather_score)
        if is_pitcher and pd.notna(combined_score):
            # A hitter-friendly environment is negative for pitchers.
            combined_score = 100.0 - float(combined_score)
        conf = park_value(profiles, home, split, 'Confidence')
        weather_conf = m.get('Weather Confidence', 1.0)
        env_conf = float(np.nanmean([conf, weather_conf])) if pd.notna(conf) else float(weather_conf or 0.0)
        rows.append({
            'player_names': name, 'team': team, 'opponent': normalize_team(r.get(ocol)) if ocol else '',
            'home_team': home, 'away_team': away, 'venue': m.get('Venue'), 'bats': bats, 'throws': throws,
            'opposing_pitcher': opposing_pitcher_name, 'opposing_pitcher_throws': opp_hand,
            'park_split': split or 'BLEND', 'contact_environment': contact, 'power_environment': power,
            'weather_score': weather_score, 'temperature': m.get('Temperature'), 'humidity': m.get('Humidity'),
            'precipitation_probability': m.get('Precipitation Probability'), 'wind_speed': m.get('Wind Speed'),
            'wind_direction': m.get('Wind Direction'), 'roof_type': m.get('Roof Type'),
            'game_environment_score': float(combined_score) if pd.notna(combined_score) else np.nan,
            'environment_confidence': env_conf,
            'is_pitcher': is_pitcher,
        })
    return pd.DataFrame(rows)


def build_team_environment_scores(players: pd.DataFrame, matchups: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    if players is None or players.empty or matchups is None or matchups.empty or profiles is None or profiles.empty:
        return pd.DataFrame()
    p = players.copy()
    norms={re.sub(r'[^a-z0-9]','',str(c).lower()):c for c in p.columns}
    tcol=norms.get('team'); bcol=norms.get('bats');
    if not tcol: return pd.DataFrame()
    rows=[]
    for _, m in matchups.iterrows():
        away, home = normalize_team(m.get('Away Team')), normalize_team(m.get('Home Team'))
        for team in (away, home):
            g=p[p[tcol].map(normalize_team)==team]
            scores=[]
            for _, r in g.iterrows():
                bats=str(r.get(bcol,'')).upper().strip() if bcol else ''
                split='LHH' if bats=='L' else ('RHH' if bats=='R' else None)
                score=park_value(profiles, home, split, 'Ballpark Score')
                if pd.notna(score): scores.append(score)
            score=float(np.mean(scores)) if scores else park_value(profiles, home, None, 'Ballpark Score')
            rows.append({'Team':team,'Away Team':away,'Home Team':home,'Venue Team':home,
                         'Game Environment Score':score,'Ballpark Coverage':1.0 if pd.notna(score) else 0.0})
    return pd.DataFrame(rows)


def fetch_mlb_handedness_map(season: int, fetch_json: Callable[[str], Any] = _fetch_json) -> dict[str, dict[str, str | None]]:
    """Fetch league-wide batter/pitcher handedness in one MLB Stats API call when available."""
    try:
        payload = fetch_json(f'https://statsapi.mlb.com/api/v1/sports/1/players?season={int(season)}')
    except Exception:
        return {}
    people = payload.get('people', []) if isinstance(payload, dict) else []
    out: dict[str, dict[str, str | None]] = {}
    for person in people:
        if not isinstance(person, dict):
            continue
        name = str(person.get('fullName') or '').strip()
        if not name:
            continue
        out[name] = {
            'bats': (person.get('batSide') or {}).get('code'),
            'throws': (person.get('pitchHand') or {}).get('code'),
        }
    return out


def _venue_weather_context(venue_id: object, game_time: object, fetch_json: Callable[[str], Any] = _fetch_json) -> dict[str, Any]:
    """Return a conservative game-time weather snapshot. Wind direction is displayed but not scored without park orientation."""
    from datetime import datetime, timezone
    try:
        venue = fetch_json(f'https://statsapi.mlb.com/api/v1/venues/{int(venue_id)}')
        venue_obj = (venue.get('venues') or [{}])[0]
        loc = venue_obj.get('location') or {}
        coords = loc.get('defaultCoordinates') or {}
        lat, lon = coords.get('latitude'), coords.get('longitude')
        roof = str((venue_obj.get('fieldInfo') or {}).get('roofType') or '')
        if lat is None or lon is None:
            return {}
        dt = datetime.fromisoformat(str(game_time).replace('Z', '+00:00')).astimezone(timezone.utc)
        day = dt.strftime('%Y-%m-%d')
        url = (
            'https://api.open-meteo.com/v1/forecast?latitude=' + str(lat) + '&longitude=' + str(lon) +
            '&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m,wind_direction_10m'
            '&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=UTC&start_date=' + day + '&end_date=' + day
        )
        weather = fetch_json(url)
        hourly = weather.get('hourly') or {}
        times = hourly.get('time') or []
        if not times:
            return {'Roof Type': roof}
        target = dt.replace(minute=0, second=0, microsecond=0).strftime('%Y-%m-%dT%H:%M')
        idx = min(range(len(times)), key=lambda i: abs(datetime.fromisoformat(times[i]).replace(tzinfo=timezone.utc) - datetime.fromisoformat(target).replace(tzinfo=timezone.utc)))
        def pick(key):
            vals = hourly.get(key) or []
            return vals[idx] if idx < len(vals) else None
        temp = pick('temperature_2m'); hum = pick('relative_humidity_2m'); precip = pick('precipitation_probability')
        wind = pick('wind_speed_10m'); wind_dir = pick('wind_direction_10m')
        fixed_roof = any(token in roof.lower() for token in ('dome', 'fixed'))
        if fixed_roof:
            score = 50.0
        else:
            t = float(temp) if temp is not None else 70.0
            h = float(hum) if hum is not None else 50.0
            p = float(precip) if precip is not None else 0.0
            score = float(np.clip(50.0 + np.clip((t - 70.0) * .65, -12, 12) + np.clip((h - 50.0) * .08, -4, 4) - np.clip(p * .04, 0, 4), 30, 70))
        return {
            'Temperature': temp, 'Humidity': hum, 'Precipitation Probability': precip,
            'Wind Speed': wind, 'Wind Direction': wind_dir, 'Roof Type': roof,
            'Weather Score': score,
            'Weather Confidence': float(np.clip(1.0 - (float(precip or 0) / 180.0), .45, 1.0)),
        }
    except Exception:
        return {}


def add_weather_to_matchups(matchups: pd.DataFrame, fetch_json: Callable[[str], Any] = _fetch_json) -> pd.DataFrame:
    if matchups is None or matchups.empty:
        return pd.DataFrame() if matchups is None else matchups.copy()
    out = matchups.copy()
    contexts = []
    for _, row in out.iterrows():
        contexts.append(_venue_weather_context(row.get('Venue ID'), row.get('Game Time'), fetch_json=fetch_json))
    weather = pd.DataFrame(contexts, index=out.index)
    return pd.concat([out, weather], axis=1)


def _combine_park_weather(park_score: float, weather_score: float | None) -> float:
    if pd.isna(park_score):
        return np.nan
    if weather_score is None or pd.isna(weather_score):
        return float(park_score)
    # Weather amplifies/suppresses the park deviation from neutral rather than replacing park research.
    weather_edge = (float(weather_score) - 50.0) / 50.0
    park_edge = float(park_score) - 50.0
    combined = 50.0 + park_edge * (1.0 + 0.35 * weather_edge) + (float(weather_score) - 50.0) * 0.30
    return float(np.clip(combined, 0.0, 100.0))
