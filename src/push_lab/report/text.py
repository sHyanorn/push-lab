"""Prose for the report and the README.

Templates hold no numbers: every placeholder is filled from the results by ``build.facts``,
and a test fails if a digit appears here outside identifiers and the interval level.
"""

SUMMARY = (
    "An offline lab for push notification recommendation: recall, ranking, uplift gating, "
    "send volume, timing and copy selection built on public data, with every result labeled "
    "by how far it can be trusted without live traffic."
)

READING = (
    "Brackets give 95% intervals. Tier A rests on randomized real data, Tier C on simulator "
    "runs with a sensitivity sweep. Settings are points drawn over the parameter box, not "
    "probabilities of the real world."
)

TIERS_HEAD = "Evidence tiers"

NO_TIER_B = "{no_tier_b}"

# Finding 1: per-day OPE and carryover
F1_TITLE = "Per-day off-policy estimates favor more pushes when sending annoys users"
F1_SCOPE = (
    "Tier C. In this simulator, along annoyance elasticities from {ope_range} with the other "
    "parameters at the reference, over {ope_settings} one-at-a-time settings."
)
F1_BODY = (
    "At the reference setting (annoyance {ref_annoyance}), per-day DR on the exploration logs "
    "has a bias in active-day share of {bias_n} for no pushes, {bias_thalf} for half the "
    "volume and {bias_t2} for two pushes a day. The bias for half the volume changes "
    "sign at an annoyance of {flip_thalf} and for no pushes at {flip_n}; above that the "
    "estimator underestimates lower volumes and overestimates two pushes a day. For the three "
    "targets that send about one push a day, as the logs do, DR is validated at "
    "{ope_daily_validated} each."
)
F1_POST_HOC = (
    "At the reference the simulator's truth puts no pushes ahead of one push a day by "
    "{ope_nt_truth} in active-day share, while DR estimates {ope_nt_dr}, the opposite sign. "
    "DR reverses this ordering at {ope_nt_reversed} settings and that of two a day against "
    "one at {ope_t2t_reversed}; its estimate of no pushes minus one a day stays between "
    "{ope_nt_dr_span} along the annoyance grid while the truth moves from {ope_nt_truth_span}."
)
F1_SHOWS = (
    "Per-day importance weights cannot see that recent volume lowers later organic visits, "
    "so in this simulator they favor higher volumes above the crossings, by an amount that "
    "grows with annoyance."
)
F1_LIMITS = (
    "The size or sign of annoyance in a real app. The sweep moves one "
    "parameter at a time; fatigue, opt-out, return and novelty changed no direction."
)

# Finding 2: delivery policies
F2_TITLE = "Whether a daily push pays turns on annoyance and the base rate"
F2_SCOPE = (
    "Tier C. {main_settings} Latin hypercube settings (main sweep), the same settings without "
    "annoyance, and a one-at-a-time grid."
)
F2_BODY = (
    "At the reference setting one push a day changes the daily active share by {ref_lift_fn} "
    "against no pushes. On the one-at-a-time grid it changes sign along two parameters only: it "
    "gains below an annoyance elasticity of {fn_flip} and above a base-rate factor of "
    "{fn_flip_base}. Across the {main_settings} main settings, its lift is below the band of "
    "published program effects ({band_low} to {band_high}) in {band_below}, inside it in "
    "{band_inside}, above it in {band_above} and straddles it in {band_straddling}."
)
F2_GATES = (
    "At equal volume the decomposed gate G2 beats the response threshold S in most settings, "
    "while the T-learner gate G is worse on average (H1). Sending at each user's best slot shows "
    "no detectable difference from one fixed slot (H4), and is slightly worse on average "
    "without annoyance."
)
F2_SHOWS = (
    "Within the swept assumptions, the sign of a daily push's effect is set by annoyance and the "
    "base rate. At equal volume, a gate that scores open probability and baseline visits "
    "separately beats a response threshold, and a T-learner uplift gate does not."
)
F2_LIMITS = (
    "Real effect sizes. Part of the G2 win comes from the simulator treating opens and organic "
    "visits as independent given the features, which G2 assumes."
)

# Finding 3: checks on randomized real data
F3_TITLE = "Checks on randomized real data, two of them against expectation"
F3_SCOPE = "Tier A. KuaiRand random exposures, Hillstrom, Criteo and the Open Bandit Dataset."
F3_BODY = (
    "The fused fine ranker ranks random exposures worse than the two-tower score it refines "
    "(C2). On Criteo the T-learner targets visits worse than a plain response model (U4). On "
    "the full Men's campaign of the Open Bandit Dataset, IPS estimates the BTS click rate at "
    "{obd_ips} of its on-policy value and DR at {obd_dr}, so no estimator passed and the "
    "project has no Tier B result."
)
F3_OBD_MISMATCH = (
    "The target is BTS with its fixed production prior, which the deployed BTS was not. Post "
    "hoc, re-weighting to the BTS logs' own frequencies moves IPS to {obd_ips_post_hoc}."
)
F3_SHOWS = (
    "A ranker fitted on logged impressions can rank random candidates worse than its retrieval "
    "score; where uplift grows with baseline response, a response threshold is a strong "
    "baseline; standard estimators fell short of a known policy value on real logs."
)
F3_LIMITS = (
    "Anything about push: the trials are video feed, e-mail and ads. The OBD ratio measures the "
    "estimators, not the value of any push policy."
)

# Templates
TPL_TITLE = "Template selection"
TPL_SCOPE = "Tier C, equal volume: every policy sends one push a day at the same slot."
TPL_BODY = (
    "The recovering bandit beats uniform templates on average (B1); the recency penalty shows "
    "no detectable effect (B2). At the reference setting the spread of results over seed "
    "triples is {b1_spread} times the bootstrap standard error for B1 and {b2_spread} for B2, "
    "so the per-setting win and loss counts are far less certain than their intervals. At the "
    "reference itself B1 averages {b1_reference} over {b1_triples} seed triples, "
    "{b1_negative} of them negative: no detectable gain."
)
TPL_LIMITS = (
    "The simulator never reads the text: copy effects are drawn per template, so nothing here "
    "says whether the generated copy is good or which wording works for real users."
)
TPL_RATING = (
    "Blind rating by {rater}: of {kept_items} kept templates, {kept_clear} clear, {kept_fits} "
    "fitting for a short-video push and {kept_misleading} misleading; of {other_items} that "
    "passed the filters but were not kept, {other_clear}, {other_fits} and "
    "{other_misleading}. The rater may have seen the catalog first. One rater cannot show "
    "that the copy does not mislead readers."
)

SHOWS = "What it shows"
LIMITS = "What it does not show"
POST_HOC = "post hoc"

MATRIX_TITLE = "Evidence matrix"
MATRIX_NOTE = "Every claim with its tier, value, interval and the results file it comes from."

STORY_TITLE = "Core story, as registered"

# (SPEC Section 3 item, outcome) pairs; outcomes cite evidence ids and decisions
STORY = (
    (
        "Sending the top item to every user every day wins on clicks in week one",
        "Not tested: no comparison of week-one opens was registered; the policy comparisons "
        "use active days (D034).",
    ),
    (
        "It loses users to fatigue and opt-outs by week four",
        "Partly (F-N, D034). At the reference a daily push changes the daily active share by "
        "{ref_lift_fn}. On the one-at-a-time grid the sign turns on annoyance (a gain below "
        "{fn_flip}) and the base rate (a gain above {fn_flip_base}), never on fatigue or opt-out.",
    ),
    (
        "An uplift gate under a volume budget, at each user's active slot, with a bandit "
        "choosing copy, sends fewer pushes and keeps more users active",
        "Not as stated (H1 to H4, B1, B2; D034, D040). The T-learner gate loses to a response "
        "threshold at equal volume; the decomposed gate G2 wins. At half volume the gate beats a "
        "daily push in {h3_wins} of {main_settings} main settings and in {h3_af_wins} without "
        "annoyance. The best slot adds nothing detectable. The bandit beats uniform templates on "
        "average; the recency penalty has no detectable effect.",
    ),
    (
        "The conclusion holds across most plausible settings, and the report shows where it flips",
        "Partly (F-N, H3; D034). Flips are located for F-N along annoyance at {fn_flip} and "
        "the base rate at {fn_flip_base}, and for H3 along annoyance at {h3_flip}; every other "
        "crossing of H1 to H4 is unresolved.",
    ),
    (
        "Every number carries an evidence tier",
        "Holds, with no Tier B result (D046).",
    ),
)
STORY_HEAD = ("Item", "Outcome")
