
## v13.48.2 launch safety

`run_app.bat` now launches the single-upload MLB Command Center directly on **http://localhost:8528**. This avoids accidentally landing on an older Streamlit app that may still be running on the default port 8501. The page title and sidebar both display **v13.48.2 · SINGLE-UPLOAD BUILD** so the running version is obvious.

If you see **Slate Inputs**, **Use included sample files**, or six separate upload boxes, you are looking at an older app/tab rather than this build.

# Drew DFS Command Center v13.48.2 — Single-Upload MLB Slate Intelligence

This build updates the MLB workflow around the current research sheets supplied for the rebuild.

## New normal workflow

1. Open **Command Center**.
2. Use **Upload MLB Slate Files** and select all slate CSVs at once.
3. The app classifies each file from its columns.
4. Review the **Slate File Detection** panel.
5. Override any ambiguous file under **Review / override file classifications**.
6. Click **Analyze MLB Slate**.

There are **no separate manual upload slots** in this build. Upload all slate files once. If a file is ambiguous or misclassified, correct its type with the classification dropdown beside that already-uploaded file.

## Auto-detected file types

- `SP1, SP2, C, 1B, 2B, 3B, SS, OF...` → **Lineup Portfolio**
- `HardperSwing` → **Team HardPerSwing**
- `Weighted True AVG` → **Starter Weakness**
- `names + avgScore + teamOwnPct + oppSP` → **Scoring Sheet**
- `xwOBA + xSLG + xBA + BABIP + HWS Ratio` → **Bullpen Research**
- `Player + Position + Salary + Median + Ceiling + Own` → **Player ROO**
- `Weighted HWSr + HWSr handedness fields` → **Weighted HWSr**

Weak or ambiguous matches are labeled **Needs classification** instead of being silently assigned.

## Stack Edge

The **Scoring Sheet is the primary stack signal**. The model adds optional research only when that source exists.

Primary blend when every source is available:

- 40% Scoring Sheet Strength
- 20% Hitter ROO Strength
- 15% Starter Vulnerability
- 15% Bullpen Vulnerability Score
- 5% Team HardPerSwing
- 5% Ownership Leverage

Missing components are excluded and the remaining weights are renormalized. Missing values are never silently converted to zero.

### Scoring Sheet Strength

Built from slate-relative percentiles of:

- avgScore — 35%
- eightPlusRuns — 22%
- topScore — 18%
- winPercentage — 8%
- avgFirstInning — 5%
- firstInningLeadPct — 4%
- avgFifthInning — 5%
- fifthInningLeadPct — 3%

### Hitter ROO Strength

Uses the best projected hitters from the Player Range of Outcome sheet, excluding pitcher rows:

- top-five Median
- top-five Ceiling
- average 15+ outcome probability
- median salary-adjusted value
- hitter ownership is displayed separately and stack ownership is used for leverage

## Locked Bullpen Vulnerability Score (BVS)

Every component is first converted to a 0–100 percentile among the uploaded bullpens, where **100 = most vulnerable / worst**.

`BVS = 30% xwOBA + 22% xSLG + 20% HWS Ratio + 13% xBA + 8% AVG + 7% BABIP`

Interpretation:

- 90–100 — **ELITE BULLPEN TO ATTACK**
- 80–89.9 — **VERY BAD BULLPEN**
- 70–79.9 — **STRONG STACK BOOST**
- 60–69.9 — **MILD STACK BOOST**
- 40–59.9 — **NEUTRAL**
- 25–39.9 — **ABOVE-AVERAGE BULLPEN**
- 0–24.9 — **BULLPEN DOWNGRADE**

The raw HWS Ratio and all six BVS inputs remain visible beside the composite score.

## Starting-pitcher vulnerability

Locked directions:

- **Higher Weighted HWSr = worse starter = boost to opponent**
- **Higher Weighted True AVG = worse starter = boost to opponent**

When both are uploaded, the starter-vulnerability layer can use both. When only one exists, that source carries the starter signal and confidence reflects the missing source.

## Pitcher Portfolio Analysis

The pitcher view uses Player ROO plus pitcher research and the opposing Stack Edge context to produce:

- Pitcher Quality
- Opponent Offense Score
- Matchup Risk
- DFS Value
- Pay-Up Justification
- Ownership / exposure context
- Portfolio Risk
- Chalk Risk
- Suggested Min / Target / Max exposure
- Recommendation Confidence
- CORE / KEEP / REDUCE / REMOVE CANDIDATE / AVOID

`REMOVE CANDIDATE` and `AVOID` require multiple negative signals. One bad metric cannot independently remove a pitcher.

Current Player ROO mappings include:

- Median → pitcher projection
- Ceiling → ceiling
- Own → projected ownership
- Top_finish → simulation/upside input

## Pitcher ↔ Stack Leverage

Crosses each pitcher with the offense he is actually facing. This surfaces spots where:

- the pitcher is popular or risky,
- underlying pitcher research is weak,
- the opposing stack grades well,
- and the field may be paying too much ownership for the pitcher.

This is descriptive portfolio guidance and does not automatically alter lineups.

## Manual pitcher exclusions

**Apply Suggested Pitcher Exclusions** is a preview action only.

Before any lineup pool changes, the app shows:

- pitchers being removed
- lineups affected
- lineups remaining
- stack/team exposure changes when a stack column is present

The user must manually confirm before the active lineup pool changes.

## Windows launch

1. Extract the ZIP.
2. Open the extracted `drew_dfs_command_center_v13_48_1` folder.
3. Double-click `run_app.bat`.

PowerShell:

```powershell
.\run_app.bat
```

If dependencies need to be installed separately:

```powershell
.\install_dependencies.bat
```

Manual launch:

```powershell
py -m pip install -r .\requirements.txt
py -m streamlit run .\app.py
```

## Files

```text
app.py
config.py
pages/01_Command_Center.py
modules/mlb_pitcher_portfolio.py
modules/mlb_slate_files.py
utils/ui.py
tests/test_mlb_pitcher_portfolio.py
tests/test_package_smoke.py
requirements.txt
run_app.bat
run_app.ps1
install_dependencies.bat
```

## Verification fixtures used during this rebuild

The implementation was exercised against the supplied column structures for:

- MLB Scoring Sheet
- MLB Player Range of Outcome sheet
- bullpen research / HWS Ratio sheet
- Weighted HWSr pitcher sheet

The slate CSVs themselves are not bundled into the app ZIP; you upload the current slate each time.

## v13.50 — Game Environment & Stack Allocation Intelligence

This build adds an optional **Ballpark Research** upload using the advanced LHH/RHH park-factor sheet. The app automatically attempts to resolve the selected slate through MLB schedule data, including **Away @ Home**, the home venue, probable starters, and game time. The home team is normalized to the ballpark sheet's `Stadium` team code.

### Advanced Ballpark Research

The ballpark layer uses the uploaded boost fields to build separate **Contact Environment** and **Power Environment** scores. It preserves LHH/RHH splits and uses PA as a confidence input. Contact incorporates hits, doubles, strikeout suppression, walks, xBA, xwOBA, BABIP, and AVG; Power emphasizes home runs, xSLG, xwOBA, and doubles.

### Automatic handedness

The app attempts to load MLB batter side (`L`, `R`, `S`) and pitcher throwing hand (`L`, `R`) automatically. Switch hitters resolve to the expected batting side against the opposing starter. If handedness cannot be resolved, the app uses a neutral LHH/RHH park blend rather than guessing.

### Weather interaction

Game-time weather is added when available. Temperature, humidity, precipitation risk, roof information, and wind are surfaced in the dashboard. Static park research remains usable if weather is unavailable. Network failures never block Analyze Sheets.

### Projection impact

Game Environment is a baseball-only research family in Drew Projection Intelligence. It may change **Floor, Median, and Ceiling**, with the smallest sensitivity on Floor and the largest on Ceiling. Ownership, salary, chalk, and portfolio exposure remain downstream and cannot change baseball projections.

### Suggested Team Stacks by entries

On the **Stacks** tab, enter the number of lineups you are actually playing (for example `20` for a 20-max) and choose:

- **Conservative** — research strength dominates; smaller leverage/chalk adjustments
- **Balanced** — research remains primary with meaningful ownership/chalk adjustments
- **Aggressive** — stronger movement toward research-backed leverage and away from unsupported chalk

Set a **Max Team Exposure** cap and the allocator returns exact integer lineup counts that always reconcile to the requested number of entries. The table also compares Conservative, Balanced, and Aggressive counts side by side.

## Mobile / Streamlit Cloud deployment

This package is ready for Streamlit Community Cloud. Use `app.py` as the entry point.

- Repository branch: any branch containing this package at the repository root
- Main file path: `app.py`
- Python dependencies: `requirements.txt`
- Streamlit settings: `.streamlit/config.toml`
- Uploaded slate files stay in the active Streamlit session; no Windows path is required.

On a phone, open the deployed `*.streamlit.app` URL in the browser and add it to the home screen for app-like access. The dashboard includes a mobile breakpoint that stacks columns, wraps segmented navigation, reduces page padding, and enlarges touch targets.
