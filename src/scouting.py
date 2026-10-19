"""
Player profiles and similarity search for recruitment scouting.

    from scouting import load_profiles, find_similar
    prof = load_profiles()
    find_similar(prof, "N'Golo Kanté", max_value=10e6, max_age=25)

Profiles are built from StatsBomb 2015/16 event data (see download_statsbomb.py)
and linked to Transfermarkt market values (see link_transfermarkt.py).
"""
import os
import re

import numpy as np
import pandas as pd
from unidecode import unidecode

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

POSITION_GROUP = {
    "Goalkeeper": "GK",
    "Center Back": "CB", "Left Center Back": "CB", "Right Center Back": "CB",
    "Left Back": "FB", "Right Back": "FB", "Left Wing Back": "FB", "Right Wing Back": "FB",
    "Center Defensive Midfield": "CM", "Left Defensive Midfield": "CM", "Right Defensive Midfield": "CM",
    "Center Midfield": "CM", "Left Center Midfield": "CM", "Right Center Midfield": "CM",
    "Center Attacking Midfield": "AM/W", "Left Attacking Midfield": "AM/W", "Right Attacking Midfield": "AM/W",
    "Left Wing": "AM/W", "Right Wing": "AM/W", "Left Midfield": "AM/W", "Right Midfield": "AM/W",
    "Center Forward": "ST", "Left Center Forward": "ST", "Right Center Forward": "ST", "Secondary Striker": "ST",
}
GROUP_LABEL = {"CB": "Centre-back", "FB": "Full-back / wing-back", "CM": "Central / defensive midfielder",
               "AM/W": "Attacking midfielder / winger", "ST": "Striker"}

# feature name -> (label shown on charts, family)
FEATURES = {
    "npxg_90": ("Non-penalty xG", "Shooting"),
    "shots_90": ("Shots", "Shooting"),
    "xg_per_shot": ("xG per shot", "Shooting"),
    "box_receipts_90": ("Touches in box", "Shooting"),
    "xa_90": ("Open-play xA", "Creation"),
    "key_passes_90": ("Key passes", "Creation"),
    "passes_into_box_90": ("Passes into box", "Creation"),
    "crosses_90": ("Crosses", "Creation"),
    "through_balls_90": ("Through balls", "Creation"),
    "prog_passes_90": ("Progressive passes", "Progression"),
    "prog_carries_90": ("Progressive carries", "Progression"),
    "dribbles_90": ("Dribbles attempted", "Progression"),
    "dribble_pct": ("Dribble success %", "Progression"),
    "passes_90": ("Passes attempted", "Possession"),
    "pass_pct": ("Pass completion %", "Possession"),
    "long_passes_90": ("Long passes", "Possession"),
    "long_pass_pct": ("Long pass completion %", "Possession"),
    "turnovers_90": ("Miscontrols + dispossessed", "Possession"),
    "tackles_padj_90": ("Tackles won (PAdj)", "Defending"),
    "interceptions_padj_90": ("Interceptions (PAdj)", "Defending"),
    "pressures_padj_90": ("Pressures (PAdj)", "Defending"),
    "pressures_att3_90": ("Final-third pressures", "Defending"),
    "recoveries_padj_90": ("Ball recoveries (PAdj)", "Defending"),
    "blocks_padj_90": ("Blocks (PAdj)", "Defending"),
    "clearances_padj_90": ("Clearances (PAdj)", "Defending"),
    "aerials_won_90": ("Aerial duels won", "Aerial"),
    "aerial_pct": ("Aerial win %", "Aerial"),
    "def_height": ("Defensive action height", "Defending"),
}
FEATURE_COLS = list(FEATURES)
# metrics where a lower value is better (inverted on radars only)
LOWER_IS_BETTER = {"turnovers_90"}


def norm_name(s):
    return re.sub(r"[^a-z ]", " ", unidecode(str(s)).lower()).strip()


def _shrunk_rate(success, attempts, prior_rate, k):
    """Empirical-Bayes style shrinkage of a percentage towards the population rate."""
    return 100 * (success + k * prior_rate) / (attempts + k)


def build_profiles(min_minutes=900, pm=None, keep_goalkeepers=False):
    """Aggregate player-match rows into one season profile per outfield player.

    pm: optional subset of data/player_match.parquet (e.g. half a season, for validation).
    """
    if pm is None:
        pm = pd.read_parquet(os.path.join(DATA, "player_match.parquet"))
    links = pd.read_parquet(os.path.join(DATA, "player_links.parquet"))

    # minutes per position group
    pos = pm[["player_id", "positions"]].copy()
    pos = pos.assign(p=pos.positions.str.split(";")).explode("p")
    pos[["position", "mins"]] = pos.p.str.split("=", expand=True)
    pos["mins"] = pos.mins.astype(float)
    pos["group"] = pos.position.map(POSITION_GROUP)
    gmins = pos.pivot_table(index="player_id", columns="group", values="mins", aggfunc="sum", fill_value=0)
    top_pos = pos.groupby(["player_id", "position"]).mins.sum().reset_index() \
                 .sort_values("mins").groupby("player_id").tail(1).set_index("player_id").position

    count_cols = [c for c in pm.columns if pm[c].dtype.kind in "fi" and c not in
                  ("match_id", "team_id", "player_id", "minutes", "team_poss")]
    agg = pm.groupby("player_id")[count_cols + ["minutes"]].sum()
    agg["matches"] = pm.groupby("player_id").match_id.nunique()
    agg["starts"] = pm.groupby("player_id").started.sum()
    agg["team_poss"] = pm.groupby("player_id").apply(lambda d: np.average(d.team_poss, weights=d.minutes),
                                                     include_groups=False)
    # main team / league = the one with most minutes
    main = pm.groupby(["player_id", "team", "competition"]).minutes.sum().reset_index() \
             .sort_values("minutes").groupby("player_id").tail(1).set_index("player_id")
    agg["team"] = main.team
    agg["league"] = main.competition
    names = pm.groupby("player_id").agg(player=("player", "first"), nickname=("nickname", "first"),
                                        country=("country", "first"))
    agg = agg.join(names)
    agg["display_name"] = agg.nickname.fillna(agg.player)
    agg["position"] = top_pos
    agg["group"] = gmins.drop(columns="GK", errors="ignore").idxmax(axis=1)
    agg.loc[gmins.idxmax(axis=1) == "GK", "group"] = "GK"
    agg["group_share"] = gmins.max(axis=1) / gmins.sum(axis=1)

    agg = agg[agg.minutes >= min_minutes].copy()
    if not keep_goalkeepers:
        agg = agg[agg.group != "GK"].copy()
    m90 = agg.minutes / 90
    for c in ["npxg", "shots", "np_goals", "box_receipts", "xa", "key_passes", "assists", "passes_into_box",
              "crosses", "through_balls", "prog_passes", "prog_carries", "dribbles", "passes",
              "long_passes", "pressures_att3", "aerials_won"] + \
             [k + "_padj" for k in ["tackles", "interceptions", "pressures", "recoveries", "blocks", "clearances"]] + \
             ["tackles", "interceptions", "pressures", "recoveries", "blocks", "clearances", "carries"]:
        agg[c + "_90"] = agg[c] / m90
    agg["turnovers_90"] = (agg.miscontrols + agg.dispossessed) / m90
    agg["xg_per_shot"] = (agg.npxg + 0.1 * 3) / (agg.shots + 3)  # shrink towards 0.10 with 3 pseudo-shots
    agg["pass_pct"] = 100 * agg.passes_cmp / agg.passes
    agg["dribble_pct"] = _shrunk_rate(agg.dribbles_cmp, agg.dribbles,
                                      agg.dribbles_cmp.sum() / agg.dribbles.sum(), 10)
    agg["long_pass_pct"] = _shrunk_rate(agg.long_passes_cmp, agg.long_passes,
                                        agg.long_passes_cmp.sum() / agg.long_passes.sum(), 10)
    aer = agg.aerials_won + agg.aerials_lost
    agg["aerial_pct"] = _shrunk_rate(agg.aerials_won, aer, agg.aerials_won.sum() / aer.sum(), 10)
    agg["def_height"] = agg.def_x_sum / agg.def_n

    # Transfermarkt link (main club record)
    lk = links[links.accepted].sort_values("sb_minutes").groupby("sb_player_id").tail(1).set_index("sb_player_id")
    for c in ["tm_player_id", "tm_name", "date_of_birth", "age_2016", "value_2016_eur", "value_2017_eur",
              "value_2018_eur", "value_2019_eur", "peak_value_after_2016_eur", "foot", "height_in_cm",
              "tm_position", "tm_sub_position", "tm_url"]:
        agg[c] = lk[c].reindex(agg.index)
    agg["value_m"] = agg.value_2016_eur / 1e6
    agg["age"] = agg.age_2016
    return agg.reset_index()


def load_profiles(min_minutes=900, cache=True):
    path = os.path.join(DATA, f"profiles_{min_minutes}.parquet")
    if cache and os.path.exists(path):
        return pd.read_parquet(path)
    prof = build_profiles(min_minutes)
    prof.to_parquet(path, index=False)
    return prof


def standardise(prof, features=FEATURE_COLS, by_group=True):
    """z-scores of each feature, computed within position group (default) or across all players."""
    X = prof[features].astype(float)
    if by_group:
        return X.groupby(prof["group"]).transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    return (X - X.mean()) / X.std(ddof=0)


def cosine_matrix(Z):
    Z = np.asarray(Z, dtype=float)
    U = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return U @ U.T


def lookup(prof, name):
    """Find a player by (part of) his name or nickname, accent-insensitive."""
    q = norm_name(name)
    cand = prof[prof.display_name.map(norm_name).str.contains(q) | prof.player.map(norm_name).str.contains(q)
                | prof.tm_name.fillna("").map(norm_name).str.contains(q)]
    if cand.empty:
        raise KeyError(f"No player matching '{name}' with enough minutes")
    if len(cand) > 1:
        exact = cand[(cand.display_name.map(norm_name) == q) | (cand.tm_name.fillna("").map(norm_name) == q)]
        cand = exact if len(exact) else cand.sort_values("minutes", ascending=False)
    return cand.index[0]


def find_similar(prof, player, n=10, max_value=None, max_age=None, leagues=None, same_group=True,
                 min_minutes=None, Z=None):
    """
    Most similar players to `player` by cosine similarity of position-standardised per-90 profiles.

    max_value : maximum Transfermarkt value in EUR at 1 July 2016 (players with no value are dropped
                when a budget is given)
    max_age   : maximum age at 1 July 2016
    leagues   : iterable of league names to search in
    same_group: only compare with players from the same position group (recommended)
    """
    if Z is None:
        Z = standardise(prof)
    i = lookup(prof, player)
    tgt = prof.loc[i]
    U = Z.values / np.linalg.norm(Z.values, axis=1, keepdims=True)
    sim = U @ U[prof.index.get_loc(i)]
    out = prof.assign(similarity=sim)
    mask = out.index != i
    if same_group:
        mask &= out.group == tgt.group
    if max_value is not None:
        mask &= out.value_2016_eur <= max_value
    if max_age is not None:
        mask &= out.age < max_age + 1
    if leagues is not None:
        mask &= out.league.isin(list(leagues))
    if min_minutes is not None:
        mask &= out.minutes >= min_minutes
    cols = ["display_name", "team", "league", "position", "age", "value_m", "minutes", "similarity"]
    return out[mask].sort_values("similarity", ascending=False)[cols].head(n)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Find statistically similar players (StatsBomb 2015/16).")
    ap.add_argument("player")
    ap.add_argument("--max-value", type=float, help="max market value in EUR millions (summer 2016)")
    ap.add_argument("--max-age", type=int)
    ap.add_argument("--league", action="append")
    ap.add_argument("-n", type=int, default=10)
    a = ap.parse_args()
    prof = load_profiles()
    res = find_similar(prof, a.player, n=a.n, max_value=a.max_value * 1e6 if a.max_value else None,
                       max_age=a.max_age, leagues=a.league)
    t = prof.loc[lookup(prof, a.player)]
    print(f"Target: {t.display_name} ({t.team}, {t.position}, age {t.age:.0f}, value €{t.value_m:.1f}m)\n")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
