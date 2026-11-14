"""Prose for the report and the README.

Templates hold no numbers: every placeholder is filled from the results by ``facts.facts``,
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
F1_TITLE = (
    "In the simulator, per-day off-policy estimates favor more pushes when sending annoys users"
)
F1_SCOPE = (
    "Tier C. In this simulator: the registered reference run, and a sweep of {ope_settings} "
    "settings on {ope_grids} one-at-a-time grids, of which the annoyance grid runs from "
    "{ope_range} with the other parameters at the reference."
)
F1_BODY = (
    "In the registered reference run (annoyance {ref_annoyance}; D045), per-day DR on the "
    "exploration logs "
    "has a bias in active-day share of {bias_n} for no pushes, {bias_thalf} for half the "
    "volume and {bias_t2} for two pushes a day. In the sweep (D048) the bias for half the volume "
    "changes sign at an annoyance of {flip_thalf} and for no pushes at {flip_n}; above that the "
    "estimator underestimates lower volumes and overestimates two pushes a day. For the three "
    "targets that send about one push a day, as the logs do, DR is validated at "
    "{ope_daily_validated} each. Fatigue, opt-out, return and novelty change no sign."
)
F1_POST_HOC = (
    "At the reference of the sweep the simulator's truth puts no pushes ahead of one push a day by "
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
    "The size or sign of annoyance in a real app, or how parameters interact: the sweep moves "
    "one parameter at a time."
)

# Finding 2: delivery policies
F2_TITLE = "In the simulator, whether a daily push pays turns on annoyance and the base rate"
F2_SCOPE = (
    "Tier C. {main_settings} Latin hypercube settings (main sweep), the same settings without "
    "annoyance, and a one-at-a-time grid."
)
F2_BODY = (
    "At the reference setting one push a day changes the daily active share by {ref_lift_fn} "
    "against no pushes. On the one-at-a-time grid it changes sign along two parameters only: it "
    "gains below an annoyance elasticity of {fn_flip} and above a base-rate factor of "
    "{fn_flip_base}. {crossing_note} Across the {main_settings} main settings, its lift is "
    "below the band of published program effects ({band_low} to {band_high}) in {band_below}, "
    "inside it in {band_inside}, above it in {band_above} and straddles it in "
    "{band_straddling}. Under the registered reading, one push a day adds less activity than "
    "the smallest reported program change in most settings, and conclusions that rely on push "
    "benefits are weaker for it."
)
CROSSING_NOTE = (
    "Crossing intervals cover user sampling with the grid's seeds fixed, not the interpolation "
    "or model-fit randomness (D034)."
)
F2_GATES = (
    "At equal volume the decomposed gate G2 beats the response threshold S in most settings, "
    "while the T-learner gate G is worse on average (H1). Sending at each user's best slot shows "
    "no detectable difference from one fixed slot (H4; median per-setting half-width "
    "{h4_half} active days), and is slightly worse on average without annoyance. Per-setting "
    "verdicts of H1 and H3 rest on one model fit and understate fit noise (D034)."
)
F2_SHOWS = (
    "On the one-at-a-time grid, the sign of a daily push's effect is set by annoyance and the "
    "base rate. Over the sweep, at equal volume, a gate that scores open probability and "
    "baseline visits separately beats a response threshold, and a T-learner uplift gate does not."
)
F2_LIMITS = "Real effect sizes. {h2_note}"
H2_NOTE = (
    "G2 assumes an open and an organic visit are independent given the features; this holds in "
    "the simulator by construction, so part of any G2 win comes from the simulator's structure."
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
    "hoc, re-weighting to the BTS logs' own frequencies moves IPS to {obd_ips_post_hoc}, a "
    "paired change of {obd_ips_change} whose interval includes zero."
)
F3_OBD_BLOCKS = (
    "These intervals resample impressions, which are not independent in these logs. Post hoc, "
    "with ten-minute blocks as resampling units the interval of IPS is {obd_ips_blocks} and "
    "that of DR {obd_dr_blocks}; the registered rule would pass both, IPS in {obd_ips_streams} "
    "of {obd_streams} other random streams and DR in {obd_dr_streams}. The registered verdict "
    "stands, and no estimator is treated as validated (D063)."
)
F3_SHOWS = (
    "A ranker fitted on logged impressions can rank random candidates worse than its retrieval "
    "score; where uplift grows with baseline response, a response threshold is a strong "
    "baseline; standard estimators came out about {obd_shortfall} below a known policy value on "
    "real logs; that value lies outside their registered intervals over impressions and, DM "
    "apart, inside their post hoc intervals over ten-minute blocks (D063); these data do not "
    "show where the gap comes from."
)
F3_LIMITS = (
    "Anything about push: the data are video feed, e-mail, ads and e-commerce recommendations. "
    "The OBD ratio measures the "
    "estimators, not the value of any push policy."
)

# Templates
TPL_TITLE = "Template selection"
TPL_SCOPE = (
    "Tier C, in the simulator, equal volume: every policy sends one push a day at the same slot. "
    "Against no pushes, one push a day changes the daily active share by {ref_lift_fn} at the "
    "reference (D034)."
)
TPL_BODY = (
    "The recovering bandit beats uniform templates on average (B1). The recency penalty "
    "depends on the setting by the registered rule and shows no detectable effect on average "
    "(B2). At the reference setting the spread of results over seed "
    "triples is {b1_spread} times the bootstrap standard error for B1 and {b2_spread} for B2, "
    "so the per-setting win and loss counts are far less certain than their intervals. At the "
    "reference itself B1 averages {b1_reference} over {b1_triples} seed triples, "
    "{b1_negative} of them negative: no detectable gain."
)
TPL_LIMITS = (
    "The simulator never reads the text: copy effects are drawn per template, so nothing here "
    "says whether the generated copy is good or which wording works for real users."
)
RATING_TAG = "descriptive"
TPL_RATING = (
    "Rating of the generated copy by {rater} (METHODS 13.3), meant to be blind, though the "
    "rater may have seen the catalog first: of {kept_items} kept templates, {kept_clear} were "
    "rated clear, {kept_fits} fitting for a short-video push and {kept_misleading} misleading; "
    "of {other_items} that passed the filters but were not kept, {other_clear}, {other_fits} "
    "and {other_misleading}. Some kept templates make claims the data cannot back (a short or "
    "quick video, a new or fresh one, popular lately; D038). One rater cannot show that such "
    "claims do not mislead readers."
)

SPREAD_NOTE = (
    "The spread over refits or seed triples was measured at the reference setting and is shown "
    "on the annoyance-free rows as well."
)
LOSING = "Losing settings of the main sweep"

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
        "Holds in the simulator, as a registered secondary result without a test (METHODS "
        "12.7): a daily push at the best slot has more week-one opens per user than the "
        "half-volume gate, by point estimate, in {w1_t_above_g} of {main_settings} main "
        "settings. At the reference they are {opens_w1_t} and {opens_w1_g}, in one model fit.",
    ),
    (
        "It loses users to fatigue and opt-outs by week four",
        "Partly, in the simulator (F-N, D034). Week-four active days are higher without "
        "pushes than under a daily push, by point estimate, in {w4_n_above_f} of "
        "{main_settings} main settings and in {w4_n_above_f_af} of the {af_settings} without "
        "annoyance. At the reference, a daily push leaves "
        "{active_w4_f} active days in week four against {active_w4_n} without pushes, and "
        "changes the daily active share by {ref_lift_fn}. On the one-at-a-time grid the sign "
        "turns on annoyance (a gain below {fn_flip}) and the base rate (a gain above "
        "{fn_flip_base}), never on fatigue or opt-out.",
    ),
    (
        "An uplift gate under a volume budget, at each user's active slot, with a bandit "
        "choosing copy, sends fewer pushes and keeps more users active",
        "Not as stated, in the simulator (H1 to H4, B1, B2; D034, D040). The T-learner gate "
        "loses to a response threshold at equal volume; the decomposed gate G2 wins, partly "
        "through the simulator's structure, which G2 assumes. At half volume the gate beats a "
        "daily push in {h3_wins} of {main_settings} main settings and loses in {h3_losses} "
        "(per-setting verdicts understate fit noise), and wins in {h3_af_wins} without "
        "annoyance. The best slot adds nothing detectable. The bandit beats uniform templates on "
        "average; the recency penalty has no detectable effect. The simulator does not read the "
        "copy, so this says nothing about which text works.",
    ),
    (
        "The conclusion holds across most plausible settings, and the report shows where it flips",
        "Partly (F-N, H3; D034). Flips are located for F-N along annoyance at {fn_flip} and "
        "the base rate at {fn_flip_base}, and for H3 along annoyance at {h3_flip}; every other "
        "crossing of H1 to H4 is unresolved. {crossing_note}",
    ),
    (
        "Every number carries an evidence tier",
        "Holds for the results in the evidence matrix, each Tier A or C with an interval or a "
        "verdict, and "
        "there is no Tier B result (D046; D063 for the post hoc block intervals). Counts of the "
        "generated copy and its rating are "
        "descriptive; registered verdicts later overturned are kept as history.",
    ),
)
STORY_HEAD = ("Item", "Outcome")
