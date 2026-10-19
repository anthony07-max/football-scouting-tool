"""
Lightweight Streamlit front end for the scouting tool.

    streamlit run app.py

Requires the data pipeline to have been run first (see README).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Radar

sys.path.insert(0, "src")
import scouting as sc  # noqa: E402

st.set_page_config(page_title="Football scouting tool", layout="wide")


@st.cache_data
def data():
    prof = sc.load_profiles()
    return prof, sc.standardise(prof)


prof, Z = data()
st.title("Player similarity scouting tool")
st.caption("StatsBomb Open Data, big-five leagues 2015/16 (Bundesliga partial) · Transfermarkt values at 1 July 2016 "
           "(CC0 transfermarkt-datasets) · 900+ minutes, outfield players")

names = prof.sort_values("minutes", ascending=False).display_name.tolist()
c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
target = c1.selectbox("Target player", names, index=names.index("N'Golo Kanté"))
max_value = c2.number_input("Max value (€m, 0 = no limit)", 0.0, 200.0, 10.0, 1.0)
max_age = c3.number_input("Max age (0 = no limit)", 0, 40, 27)
leagues = c4.multiselect("Leagues", sorted(prof.league.unique()), default=sorted(prof.league.unique()))

ti = prof.index[prof.display_name == target][0]
t = prof.loc[ti]
st.markdown(f"**{t.display_name}**, {t.team} · {t.position} · age {t.age:.0f} · €{t.value_m:.1f}m · "
            f"{t.minutes:.0f} minutes")
res = sc.find_similar(prof, target, n=15, max_value=max_value * 1e6 or None, max_age=max_age or None,
                      leagues=leagues, Z=Z)
st.dataframe(res.rename(columns={"display_name": "player", "value_m": "value €m"}).style.format(
    {"age": "{:.0f}", "value €m": "{:.1f}", "minutes": "{:.0f}", "similarity": "{:.2f}"}),
    width="stretch", hide_index=True)

if len(res):
    pick = st.selectbox("Compare on a radar", res.display_name.tolist())
    mi = res.index[res.display_name == pick][0]
    peers = prof[prof.group == t.group]
    feats = [f for f in sc.FEATURE_COLS if f != "def_height"]

    def pct(i):
        v = [(peers[f] < prof.loc[i, f]).mean() * 100 for f in feats]
        return [100 - x if f in sc.LOWER_IS_BETTER else x for x, f in zip(v, feats)]

    radar = Radar([sc.FEATURES[f][0] for f in feats], [0] * len(feats), [100] * len(feats), num_rings=4,
                  ring_width=1, center_circle_radius=1)
    fig, ax = radar.setup_axis(figsize=(8, 8))
    radar.draw_circles(ax=ax, facecolor="#f3f3f3", edgecolor="#d0d0d0")
    radar.draw_radar_compare(pct(ti), pct(mi), ax=ax, kwargs_radar={"facecolor": "#1f4e79", "alpha": 0.55},
                             kwargs_compare={"facecolor": "#f39c12", "alpha": 0.55})
    radar.draw_param_labels(ax=ax, fontsize=9)
    ax.set_title(f"Blue: {t.display_name}   Orange: {pick}\n(percentile vs {sc.GROUP_LABEL[t.group].lower()}s)")
    st.pyplot(fig, width="content")
