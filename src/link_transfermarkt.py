"""
Link StatsBomb 2015/16 players to the CC0 Transfermarkt dataset
(https://github.com/dcaribou/transfermarkt-datasets) to attach date of birth,
market value in summer 2016 and later market values.

1. Clubs are matched by fixture dates: a StatsBomb team and a Transfermarkt club
   in the same league that played on the same dates all season are the same club.
2. Players are matched within each club's 2015/16 league squad by name similarity
   (accents stripped, nickname considered) combined with agreement in minutes
   played, using an optimal (Hungarian) assignment.

Output: data/player_links.parquet
Usage:  python src/link_transfermarkt.py   (after download_statsbomb.py and download_transfermarkt.py)
"""
import os
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from unidecode import unidecode

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TM = os.path.join(DATA, "raw", "tm")
LEAGUE_CODE = {"Premier League": "GB1", "La Liga": "ES1", "Serie A": "IT1",
               "Bundesliga": "L1", "Ligue 1": "FR1"}
REF_DATE = pd.Timestamp("2016-07-01")
# Known names with no textual overlap between the two sources
ALIASES = {"Javier Hernández Balcázar": "Chicharito"}  # start of the 2016 summer window


def norm(s):
    if not isinstance(s, str):
        return ""
    s = unidecode(s).lower()
    return re.sub(r"[^a-z ]", " ", s).strip()


def name_sim(sb_names, tm_name):
    t = norm(tm_name)
    best = 0
    for n in sb_names:
        n = norm(n)
        if not n:
            continue
        # token_set handles "Pedro Eliezer Rodriguez Ledesma" vs "Pedro";
        # token_sort guards against short-name false positives
        best = max(best, 0.6 * fuzz.token_set_ratio(n, t) + 0.4 * fuzz.token_sort_ratio(n, t),
                   fuzz.ratio(n, t))
        # surname match bonus
        if t.split() and n.split() and t.split()[-1] in n.split():
            best = max(best, 85)
    return best


def main():
    pm = pd.read_parquet(os.path.join(DATA, "player_match.parquet"))
    pm["match_date"] = pd.to_datetime(pm.match_date)
    games = pd.read_csv(os.path.join(TM, "games.csv.gz"), parse_dates=["date"])
    games = games[(games.season == 2015) & games.competition_id.isin(LEAGUE_CODE.values())]
    apps = pd.read_csv(os.path.join(TM, "appearances.csv.gz"), parse_dates=["date"])
    apps = apps[apps.game_id.isin(games.game_id)]
    players = pd.read_csv(os.path.join(TM, "players.csv.gz"), parse_dates=["date_of_birth"])
    vals = pd.read_csv(os.path.join(TM, "player_valuations.csv.gz"), parse_dates=["date"])

    # ---- 1. clubs by fixtures ---------------------------------------------------
    # For every StatsBomb match, find the Transfermarkt game in the same league on
    # the same date whose two club names best match the pair; each team is then
    # mapped to the club it was matched to most often.
    stop = {"fc", "ac", "cf", "sc", "afc", "us", "ss", "as", "de", "calcio", "club", "rc", "cd", "ud", "sd", "rcd"}

    def club_sim(a, b):
        ta = " ".join(w for w in norm(a).split() if w not in stop)
        tb = " ".join(w for w in norm(b).split() if w not in stop)
        return max(fuzz.token_set_ratio(ta, tb), fuzz.partial_ratio(ta, tb))

    votes = {}
    sb_matches = pm.groupby(["competition", "match_id"]).agg(date=("match_date", "first"),
                                                             teams=("team", lambda s: sorted(set(s))))
    for (comp, mid), r in sb_matches.iterrows():
        a, b = r.teams
        g = games[(games.competition_id == LEAGUE_CODE[comp]) & (games.date == r.date)]
        best, best_s = None, -1
        for gg in g.itertuples():
            for h, w in [(a, b), (b, a)]:
                sc = club_sim(h, gg.home_club_name) + club_sim(w, gg.away_club_name)
                if sc > best_s:
                    best_s, best = sc, {h: gg.home_club_id, w: gg.away_club_id}
        for team, cid in (best or {}).items():
            votes.setdefault((comp, team), []).append(cid)
    club_map = {k: pd.Series(v).mode()[0] for k, v in votes.items()}
    clubs = pd.Series(club_map, name="tm_club_id")

    # ---- 2. players within club squads ---------------------------------------
    sb_tot = (pm.groupby(["competition", "team", "player_id"])
                .agg(player=("player", "first"), nickname=("nickname", "first"), minutes=("minutes", "sum"))
                .reset_index())
    tm_tot = (apps.groupby(["player_club_id", "player_id"])
                  .agg(tm_name=("player_name", "first"), tm_minutes=("minutes_played", "sum"))
                  .reset_index())
    links = []
    for (comp, team), grp in sb_tot.groupby(["competition", "team"]):
        cid = club_map[(comp, team)]
        cand = tm_tot[tm_tot.player_club_id == cid].reset_index(drop=True)
        if cand.empty:
            continue
        S = np.zeros((len(grp), len(cand)))
        N = np.zeros_like(S)
        for i, r in enumerate(grp.itertuples()):
            for j, c in enumerate(cand.itertuples()):
                ns = name_sim([r.player, r.nickname, ALIASES.get(r.player)], c.tm_name)
                ms = 100 * (1 - abs(r.minutes - c.tm_minutes) / max(r.minutes, c.tm_minutes, 1))
                N[i, j] = ns
                S[i, j] = 0.75 * ns + 0.25 * ms
        rows, cols = linear_sum_assignment(-S)
        for i, j in zip(rows, cols):
            r, c = grp.iloc[i], cand.iloc[j]
            links.append(dict(competition=comp, team=team, sb_player_id=r.player_id, player=r.player,
                              nickname=r.nickname, sb_minutes=r.minutes, tm_player_id=c.player_id,
                              tm_name=c.tm_name, tm_minutes=c.tm_minutes, name_score=N[i, j],
                              score=S[i, j], tm_club_id=cid))
    links = pd.DataFrame(links)
    ratio = np.minimum(links.sb_minutes, links.tm_minutes) / np.maximum(links.sb_minutes, links.tm_minutes)
    links["accepted"] = (links.name_score >= 90) | (
        (links.name_score >= 70) & ((ratio >= 0.5) | (links.competition == "Bundesliga")))

    # ---- 3. attach profile and valuations ------------------------------------
    links = links.merge(players[["player_id", "name", "date_of_birth", "position", "sub_position", "foot",
                                 "height_in_cm", "country_of_citizenship"]]
                        .rename(columns={"player_id": "tm_player_id", "position": "tm_position",
                                         "sub_position": "tm_sub_position"}), on="tm_player_id", how="left")

    def value_at(date, max_age_days=400):
        v = vals[vals.date <= date].sort_values("date").groupby("player_id").tail(1)
        v = v[(date - v.date).dt.days <= max_age_days]
        return v.set_index("player_id")

    v16 = value_at(REF_DATE)
    links["value_2016_eur"] = links.tm_player_id.map(v16.market_value_in_eur)
    links["value_2016_date"] = links.tm_player_id.map(v16.date)
    for yr in [2017, 2018, 2019]:
        links[f"value_{yr}_eur"] = links.tm_player_id.map(value_at(pd.Timestamp(f"{yr}-07-01")).market_value_in_eur)
    later = vals[vals.date > REF_DATE].groupby("player_id").market_value_in_eur.max()
    links["peak_value_after_2016_eur"] = links.tm_player_id.map(later)
    links["age_2016"] = (REF_DATE - links.date_of_birth).dt.days / 365.25
    links["tm_url"] = "https://www.transfermarkt.co.uk/-/profil/spieler/" + links.tm_player_id.astype(str)
    links.to_parquet(os.path.join(DATA, "player_links.parquet"), index=False)
    clubs.reset_index().to_csv(os.path.join(DATA, "club_links.csv"), index=False)
    acc = links[links.accepted]
    print(f"{len(links):,} StatsBomb player-club records; {len(acc):,} linked "
          f"({acc.sb_minutes.sum() / links.sb_minutes.sum():.1%} of minutes); "
          f"{acc.value_2016_eur.notna().sum():,} with a summer-2016 value")


if __name__ == "__main__":
    main()
