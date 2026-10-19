"""
Download StatsBomb Open Data (https://github.com/statsbomb/open-data) for the
2015/16 big-five league seasons and aggregate every match into one row per
player per match (minutes, positions played and counts of on-ball and
defensive actions).

Raw event files are ~3-4 MB per match, so they are processed on the fly and
never stored; the output is data/player_match.parquet (~a few MB).
Data (c) StatsBomb, used under their open-data licence.

Usage:  python src/download_statsbomb.py
"""
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = os.path.join(os.path.dirname(__file__), "..", "data")

COMPETITIONS = [
    (2, 27, "Premier League"),
    (11, 27, "La Liga"),
    (12, 27, "Serie A"),
    (9, 27, "Bundesliga"),
    (7, 27, "Ligue 1"),
]
SET_PIECES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
GOAL = (120.0, 40.0)
session = requests.Session()


def get_json(url, retries=5):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise


def clock(s):
    m, sec = s.split(":")
    return int(m) + int(sec) / 60


def dist_goal(loc):
    return math.hypot(GOAL[0] - loc[0], GOAL[1] - loc[1])


def in_box(loc):
    return loc[0] >= 102 and 18 <= loc[1] <= 62


def progressive(start, end):
    """Ball moved at least 25% closer to the centre of the goal and at least 10 yards forward-ish."""
    d0, d1 = dist_goal(start), dist_goal(end)
    return d1 <= 0.75 * d0 and (d0 - d1) >= 5


def process_match(match, comp_label):
    mid = match["match_id"]
    events = get_json(f"{BASE}/events/{mid}.json")
    lineups = get_json(f"{BASE}/lineups/{mid}.json")

    # ---- match length (cumulative match clock) --------------------------------
    period_end = defaultdict(float)
    for e in events:
        t = e["minute"] + e["second"] / 60
        period_end[e["period"]] = max(period_end[e["period"]], t)
    last_period = max(p for p in period_end if p <= 4)
    match_end = period_end[last_period]

    # ---- team possession proxy: share of passes --------------------------------
    team_passes = defaultdict(int)
    for e in events:
        if e["type"]["name"] == "Pass":
            team_passes[e["team"]["id"]] += 1
    tot = sum(team_passes.values()) or 1

    # shot xG lookup for expected assists
    shot_xg = {}
    for e in events:
        if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty":
            kp = e["shot"].get("key_pass_id")
            if kp:
                shot_xg[kp] = e["shot"].get("statsbomb_xg", 0.0)

    rows = {}
    team_ids = [t["team_id"] for t in lineups]
    team_name = {t["team_id"]: t["team_name"] for t in lineups}
    for t in lineups:
        tid = t["team_id"]
        opp = [x for x in team_ids if x != tid][0]
        poss = team_passes[tid] / tot
        for p in t["lineup"]:
            if not p["positions"]:
                continue
            # a few lineups keep a sent-off player on the pitch; cap at the red card
            off = [clock(c["time"]) for c in p.get("cards", [])
                   if c["card_type"] in ("Red Card", "Second Yellow")]
            cap = min(off) if off else match_end
            pos_min = defaultdict(float)
            for s in p["positions"]:
                a = clock(s["from"])
                b = min(clock(s["to"]) if s["to"] else match_end, cap)
                pos_min[s["position"]] += max(0.0, b - a)
            minutes = sum(pos_min.values())
            if minutes <= 0:
                continue
            rows[p["player_id"]] = dict(
                match_id=mid, competition=comp_label, match_date=match["match_date"],
                team_id=tid, team=team_name[tid], opponent=team_name[opp],
                player_id=p["player_id"], player=p["player_name"],
                nickname=p.get("player_nickname"), country=(p.get("country") or {}).get("name"),
                minutes=minutes, team_poss=poss,
                started=p["positions"][0]["start_reason"] == "Starting XI",
                positions=dict(pos_min),
                **{k: 0.0 for k in COUNT_COLS},
            )

    for e in events:
        pl = e.get("player")
        if not pl or pl["id"] not in rows:
            continue
        r = rows[pl["id"]]
        typ = e["type"]["name"]
        loc = e.get("location")
        opp_poss = 1 - r["team_poss"]
        padj = 0.5 / max(opp_poss, 0.2)  # scale defensive actions to a 50% possession game

        def defend(key):
            r[key] += 1
            r[key + "_padj"] += padj
            if loc:
                r["def_x_sum"] += loc[0]
                r["def_n"] += 1

        if typ == "Pass":
            pa = e["pass"]
            ptype = pa.get("type", {}).get("name")
            complete = "outcome" not in pa
            end = pa["end_location"]
            if pa.get("aerial_won"):
                r["aerials_won"] += 1
            if ptype in SET_PIECES:
                if pa.get("shot_assist") or pa.get("goal_assist"):
                    r["key_passes_sp"] += 1
                continue
            r["passes"] += 1
            r["passes_cmp"] += complete
            if pa.get("length", 0) >= 32:  # ~30 m
                r["long_passes"] += 1
                r["long_passes_cmp"] += complete
            if pa.get("cross"):
                r["crosses"] += 1
                r["crosses_cmp"] += complete
            if pa.get("through_ball") or pa.get("technique", {}).get("name") == "Through Ball":
                r["through_balls"] += 1
            if complete and progressive(loc, end) and loc[0] >= 30:
                r["prog_passes"] += 1
            if complete and loc[0] < 80 <= end[0]:
                r["passes_final_third"] += 1
            if complete and in_box(end) and not in_box(loc):
                r["passes_into_box"] += 1
            if pa.get("shot_assist") or pa.get("goal_assist"):
                r["key_passes"] += 1
            if pa.get("goal_assist"):
                r["assists"] += 1
            r["xa"] += shot_xg.get(e["id"], 0.0)
        elif typ == "Carry":
            end = e["carry"]["end_location"]
            if loc and math.hypot(end[0] - loc[0], end[1] - loc[1]) >= 5:
                r["carries"] += 1
                if progressive(loc, end) and loc[0] >= 30:
                    r["prog_carries"] += 1
                if in_box(end) and not in_box(loc):
                    r["carries_into_box"] += 1
                if loc[0] < 80 <= end[0]:
                    r["carries_final_third"] += 1
        elif typ == "Ball Receipt*":
            if "outcome" not in e.get("ball_receipt", {}):
                r["receipts"] += 1
                if loc and in_box(loc):
                    r["box_receipts"] += 1
        elif typ == "Dribble":
            r["dribbles"] += 1
            if e["dribble"].get("outcome", {}).get("name") == "Complete":
                r["dribbles_cmp"] += 1
        elif typ == "Shot":
            sh = e["shot"]
            if sh["type"]["name"] == "Penalty":
                r["pens"] += 1
                r["pen_goals"] += sh["outcome"]["name"] == "Goal"
                continue
            r["shots"] += 1
            r["npxg"] += sh.get("statsbomb_xg", 0.0)
            r["np_goals"] += sh["outcome"]["name"] == "Goal"
            r["shots_on_target"] += sh["outcome"]["name"] in ("Goal", "Saved", "Saved To Post")
            if sh.get("body_part", {}).get("name") == "Head":
                r["headed_shots"] += 1
            if sh.get("aerial_won"):
                r["aerials_won"] += 1
        elif typ == "Duel":
            d = e["duel"]
            dt = d.get("type", {}).get("name")
            if dt == "Aerial Lost":
                r["aerials_lost"] += 1
            elif dt == "Tackle":
                r["tackles_att"] += 1
                if d.get("outcome", {}).get("name") in ("Won", "Success", "Success In Play", "Success Out"):
                    defend("tackles")
        elif typ == "Interception":
            if e["interception"].get("outcome", {}).get("name") not in ("Lost", "Lost In Play", "Lost Out"):
                defend("interceptions")
        elif typ == "Ball Recovery":
            if not e.get("ball_recovery", {}).get("recovery_failure"):
                defend("recoveries")
        elif typ == "Pressure":
            defend("pressures")
            if loc and loc[0] >= 80:
                r["pressures_att3"] += 1
        elif typ == "Block":
            defend("blocks")
        elif typ == "Clearance":
            defend("clearances")
            if e.get("clearance", {}).get("aerial_won"):
                r["aerials_won"] += 1
        elif typ == "Miscontrol":
            r["miscontrols"] += 1
            if e.get("miscontrol", {}).get("aerial_won"):
                r["aerials_won"] += 1
        elif typ == "Dispossessed":
            r["dispossessed"] += 1
        elif typ == "Foul Committed":
            r["fouls"] += 1
        elif typ == "Foul Won":
            r["fouls_won"] += 1
        elif typ == "Dribbled Past":
            r["dribbled_past"] += 1

    out = []
    for r in rows.values():
        pos = r.pop("positions")
        r["positions"] = ";".join(f"{k}={v:.1f}" for k, v in pos.items())
        out.append(r)
    return out


DEF_KEYS = ["tackles", "interceptions", "recoveries", "pressures", "blocks", "clearances"]
COUNT_COLS = (
    ["passes", "passes_cmp", "long_passes", "long_passes_cmp", "crosses", "crosses_cmp",
     "through_balls", "prog_passes", "passes_final_third", "passes_into_box", "key_passes",
     "key_passes_sp", "assists", "xa", "carries", "prog_carries", "carries_into_box",
     "carries_final_third", "receipts", "box_receipts", "dribbles", "dribbles_cmp", "shots",
     "npxg", "np_goals", "shots_on_target", "headed_shots", "pens", "pen_goals", "aerials_won",
     "aerials_lost", "tackles_att", "pressures_att3", "miscontrols", "dispossessed", "fouls",
     "fouls_won", "dribbled_past", "def_x_sum", "def_n"]
    + DEF_KEYS + [k + "_padj" for k in DEF_KEYS]
)


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for cid, sid, label in COMPETITIONS:
        ms = get_json(f"{BASE}/matches/{cid}/{sid}.json")
        print(f"{label} 2015/16: {len(ms)} matches", flush=True)
        jobs += [(m, label) for m in ms]
    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process_match, m, lab): m["match_id"] for m, lab in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows += f.result()
            except Exception as exc:  # pragma: no cover
                failed.append((futs[f], repr(exc)))
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} matches processed", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "player_match.parquet"), index=False)
    print(f"Saved {len(df):,} player-match rows from {df.match_id.nunique():,} matches; failed: {failed}")


if __name__ == "__main__":
    sys.exit(main())
