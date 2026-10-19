# CV bullets and LinkedIn post: scouting tool project

All figures below come from the executed notebook (`scouting_tool.ipynb`) and `results/`.

## CV bullet points (choose 2-3)

**Player Recruitment & Similarity Scouting Tool | Python, scikit-learn, UMAP, mplsoccer, Streamlit** · github.com/anthony07-max/football-scouting-tool

- Built a position-aware player similarity tool from 1,551 matches of StatsBomb event data (big-five leagues 2015/16). I engineered 28 per-90 metrics, including possession-adjusted defending, for 1,275 players and linked 99.9% of player minutes to Transfermarkt market values with fixture-based club matching and name-plus-minutes player matching.
- Validated the similarity measure with a split-half re-identification test: a player's profile from half his matches found his own profile from the other half first 36.7% of the time and in the top 5 62.7% of the time, against 0.4% by chance. It beat Euclidean, PCA and all-player-standardised alternatives.
- Back-tested recommendations against the summer 2016 transfer window. The tool matched Kanté to Idrissa Gueye (similarity 0.73, €9m), whom Everton signed for €8.5m. Across 85 targets valued at €25m+, players it picked as cheap lookalikes saw ×1.27 higher value growth by 2018 than other players who fitted the same brief, after controlling for value, age and minutes (95% CI ×1.05 to ×1.56).

*Shorter one-line version:* Built a player similarity scouting tool on StatsBomb event data and Transfermarkt values (1,275 players), validated with a split-half test (62.7% top-5 re-identification vs 0.4% chance) and back-tested on the 2016 window, where its cheap lookalikes outgrew comparable players' market values.

## LinkedIn post

"Find me someone who plays like him, but cheaper." That's one of the most common questions a club's recruitment team gets, so for my second football analytics project I built a tool to answer it. ⚽🔎

Using StatsBomb's free event data for the 2015/16 Premier League, La Liga, Serie A and Ligue 1 (1,551 matches), I built per-90 profiles for 1,275 players and linked them to Transfermarkt market values and ages.

What I did:
🔹 28 metrics covering shooting, chance creation, progression, possession, possession-adjusted defending and aerial duels
🔹 Compared each player only with others in his position group, using cosine similarity, and filtered by budget, age and league
🔹 Clustered players into 9 data-driven roles, such as ball-playing vs stopper centre-backs and deep-lying vs creative playmakers
🔹 Tested it on the summer 2016 transfer window, because we now know what happened next

What I found:
✅ The profiles are stable: a player's profile from half his matches picks out his own profile from the other half in the top 5 62.7% of the time (chance: 0.4%)
✅ Kanté's closest match under €10m was Idrissa Gueye (similarity 0.73), whom Everton signed for €8.5m that August
✅ Pogba's closest matches under €20m included André Gomes and Piotr Zieliński, who joined Barcelona and Napoli that summer
✅ Across 85 stars valued at €25m+, the tool's cheap lookalikes rose more in market value by 2018 than the rest of the eligible pool for 72% of targets

The project includes radar charts, a role map, a find_similar() function and a small Streamlit app. Notebook and write-up are on GitHub: https://github.com/anthony07-max/football-scouting-tool

Thanks to StatsBomb and the transfermarkt-datasets project for making this data freely available. I'm looking for junior data and sports analytics roles in London, so I'd love to connect with anyone working in football recruitment or data.

#FootballAnalytics #SportsAnalytics #DataScience #Recruitment #Scouting #Python #StatsBomb
