"""Prose for the report and the README.

Templates hold no numbers: every placeholder is filled from the results by ``facts.facts``,
and a test fails if a digit appears here outside identifiers and the interval level.
"""

DESCRIPTION = (
    "An offline lab for push notification recommendation: recall, ranking, uplift gating, "
    "send volume, timing and template selection built on public data, with every result labeled "
    "by how far it can be trusted without live traffic."
)
# The summary at the top of the README and of the report
SUMMARY = (
    "An offline lab for push notification recommendation on public data. Each result carries an "
    "evidence tier: A, randomized real data; B, a validated off-policy estimate; C, simulator "
    "runs with a sensitivity sweep. On Duolingo's randomized notification logs, the template "
    "bandit raises two-hour completion per notification by {dl3a} over uniform choice, and its "
    "recency penalty adds no detectable value (Tier A). In the simulator, at the reference "
    "setting a daily push changes the daily active share by {ref_lift_fn} in relative terms "
    "against no pushes, and the estimator SMIS favors one push a day over none at "
    "{hz_e1_reversed} settings, the reverse of the truth (Tier C). The registered claim that an "
    "uplift gate under a volume budget keeps more users active does not hold as stated. The lab "
    "has no Tier B result, and the simulator gives no real-world effect sizes."
)

TIERS_HEAD = "Evidence tiers"

NO_TIER_B = "{no_tier_b}"

# Duolingo notification logs
DL_TITLE = (
    "On real notification logs, a template bandit beats uniform choice and its recency penalty "
    "adds nothing detectable"
)
DL_SCOPE = (
    "Tier A. Duolingo practice reminders, published by Yancey and Settles: {dl_rows} "
    "notifications over {dl_days} days, each with its template drawn uniformly from the "
    "eligible ones. The value is that of one notification under a template policy given the "
    "logged histories (METHODS 15, D080)."
)
DL_BODY = (
    "The paper's offline lift lies inside the conservative IPS interval for {dl1_replicated} "
    "policies (DL1). The recovering difference softmax RD beats uniform templates U by {dl3a} "
    "in two-hour completion per notification, from {dl_mean_reward} under uniform choice "
    "(DL3a). RD minus D0, the same policy without its recency penalty, is {dl3b}: no detectable "
    "difference by IPS, the registered estimator (DL3b). On the descriptive curve, a template "
    "last sent within {dl_recent_days} days completes {dl_recent} less often than one not in "
    "the history. The recency weight of the paper's form is {dl_gamma}, an interval that holds "
    "the paper's production value, and its half-life, {dl_h} days, is not determined (DL2)."
)

# Delivery policies
F2_TITLE = "In the simulator, whether a daily push pays turns on annoyance and the base rate"
F2_SCOPE = (
    "Tier C. {main_settings} Latin hypercube settings (main sweep), the same settings without "
    "annoyance, and a one-at-a-time grid. Differences are in active days per user over four weeks."
)
F2_BODY = (
    "At the reference one push a day changes the daily active share by {ref_lift_fn} in "
    "relative terms against no pushes. On the one-at-a-time grid it gains below an annoyance "
    "elasticity of {fn_flip} and above a base-rate factor of {fn_flip_base}, and changes sign "
    "along no other parameter. Over the {main_settings} main settings its lift is below the "
    "band of published program effects ({band_low} to {band_high}) in {band_below}, inside it "
    "in {band_inside}, above it in {band_above} and straddles it in {band_straddling}."
)
F2_GATES = (
    "At equal volume the decomposed gate G2 beats the response threshold S in most settings and "
    "the T-learner gate G is worse on average (H1, H2). At half volume G beats a daily push by "
    "{h3_average} active days per user on average, in {h3_wins} of {main_settings} settings and "
    "in {h3_af_wins} of the {af_settings} without annoyance (H3). Sending at each user's best "
    "slot shows no detectable difference from one fixed slot in the main sweep and is slightly "
    "worse on average without annoyance (H4)."
)

# Off-policy estimates in the simulator
F1_TITLE = (
    "In the simulator, per-day off-policy estimates favor more pushes when sending annoys users"
)
F1_SCOPE = (
    "Tier C. The registered reference run and a sweep of {ope_settings} settings on "
    "{ope_grids} one-at-a-time grids; the annoyance grid runs from {ope_range}."
)
F1_BODY = (
    "In the reference run (annoyance {ref_annoyance}; D045), per-day DR on the exploration logs "
    "has a bias in active-day share of {bias_n} for no pushes, {bias_thalf} for half the volume "
    "and {bias_t2} for two pushes a day. In the sweep (D048) the bias for half the volume "
    "changes sign at an annoyance of {flip_thalf} and for no pushes at {flip_n}; above that DR "
    "underestimates lower volumes and overestimates two pushes a day, as per-day weights cannot "
    "see that recent volume lowers later organic visits. For the three targets that send about "
    "one push a day, as the logs do, DR is validated at {ope_daily_validated} each."
)
F1_POST_HOC = (
    "At the sweep's reference the truth puts no pushes ahead of one push a day by "
    "{ope_nt_truth} in active-day share, while DR estimates {ope_nt_dr}, the opposite sign. "
    "DR reverses this ordering at {ope_nt_reversed} settings and that of two a day against "
    "one at {ope_t2t_reversed}. Along the annoyance grid the truth moves from "
    "{ope_nt_truth_span} while the DR estimate stays between {ope_nt_dr_span}."
)

# Long-horizon study, a subsection of the off-policy section
HZ_TITLE = "Long-horizon estimators and logging designs"
HZ_SCOPE = (
    "Tier C, registered in D066 and D069, run in D079: {hz_settings} settings, {hz_logs_ref} "
    "logs per design at the reference and {hz_logs} elsewhere, each on {hz_users} users, "
    "against exact values in the control and {hz_truth_runs} paired online runs of each "
    "target elsewhere."
)
HZ_BODY = (
    "At the reference the truth of no pushes minus one push a day is {hz_truth_n} in active-day "
    "share, and of two a day minus one {hz_truth_t2}. No primary estimator recovers both on any "
    "design. On the current exploration logs SMIS reverses the first, at {hz_e1_reversed} "
    "settings, every one in which pushes annoy users, and FQE the second. With weekly arms, "
    "PDIS-week and SMIS-week are validated but too wide to separate, and FQE-week is biased. "
    "With four arms, arm means are validated and recover both orderings only at "
    "{hz_arm_recovers}; on {hz_users} users their intervals are too wide, and projected to all "
    "{hz_all_users} users the half-width would be about {hz_projected}. The current logs almost "
    "never follow no pushes or two a day for a week, so the volume questions need a randomized "
    "test on more users."
)
# Templates
TPL_TITLE = "Template selection"
TPL_SCOPE = (
    "Tier C, equal volume: every policy sends one push a day at the same slot. Differences "
    "are in opens per user over four weeks."
)
TPL_BODY = (
    "The recovering bandit beats uniform templates on average (B1). The recency penalty depends "
    "on the setting by the registered rule and shows no detectable effect on average (B2). At "
    "the reference itself B1 averages {b1_reference} over {b1_triples} seed triples, "
    "{b1_negative} of them negative: no detectable gain. B2 averages {b2_reference} over the "
    "same triples, {b2_negative} of them negative, with an interval that includes zero."
)
TPL_LIMITS = (
    "The simulator never reads the text: copy effects are drawn per template, so nothing here "
    "says whether the generated copy is good or which wording works for real users."
)
DESCRIPTIVE = "descriptive"
RATING_TAG = DESCRIPTIVE
TPL_RATING = (
    "Rating of the generated copy by {rater} (METHODS 13.3): of {kept_items} kept templates, "
    "{kept_clear} were rated clear, {kept_fits} fitting for a short-video push and "
    "{kept_misleading} misleading; of {other_items} that passed the filters but were not "
    "kept, {other_clear}, {other_fits} and {other_misleading}."
)

# Checks on randomized real data
F3_TITLE = (
    "On randomized real data, two components do worse than their baselines and no estimator passes"
)
F3_SCOPE = "Tier A. KuaiRand random exposures, Hillstrom, Criteo and the Open Bandit Dataset."
F3_BODY = (
    "The fused fine ranker ranks random exposures worse than the two-tower score it refines "
    "(C2), and on Criteo the T-learner targets visits worse than a plain response model (U4); "
    "both reverse the registered expectation. Two-tower retrieval beats popularity (C1), and "
    "the T-learner beats random targeting on Hillstrom and Criteo (U1, U3) and shows no "
    "detectable difference from the response model on Hillstrom (U2). On the full Men's "
    "campaign of the Open Bandit Dataset, the four estimators put the BTS click rate about "
    "{obd_shortfall} below its on-policy value, IPS at {obd_ips} of it and DR at {obd_dr}, so "
    "no estimator passed."
)
F3_OBD_POST_HOC = (
    "Re-weighting to the BTS logs' own frequencies moves IPS to {obd_ips_post_hoc}, a paired "
    "change of {obd_ips_change} that includes zero. With ten-minute blocks as resampling units "
    "the interval of IPS is {obd_ips_blocks} and that of DR {obd_dr_blocks}, and the registered "
    "rule would pass both, IPS only marginally (IPS in {obd_ips_streams} of {obd_streams} other "
    "random streams, DR in {obd_dr_streams}); the registered verdict stands and no estimator "
    "counts as validated (D063)."
)
F3_OBD_CITED = (
    "In a separate registered study, cited from another repository and not recomputed here, "
    "block intervals for IPS on the All and Women's campaigns are {obd_cited_all} and "
    '{obd_cited_women} of their own on-policy values, labeled "no shortfall shown" (D065).'
)

CROSSING_NOTE = (
    "Crossing intervals cover user sampling with the grid's seeds fixed and leave out the "
    "interpolation and model-fit randomness (D034)."
)
H2_NOTE = (
    "G2 assumes an open and an organic visit are independent given the features; this holds in "
    "the simulator by construction, so part of any G2 win comes from the simulator's structure."
)

# Method, reproduction and limits
METHOD_TITLE = "Method"
METHOD = (
    "Tier A results use randomized real data: random exposures (KuaiRand), randomized trials "
    "(Hillstrom, Criteo), uniform template choice (Duolingo) and random logs (Open Bandit "
    "Dataset). Tier C results come from a simulator built from KuaiRand tables, each over a "
    "sensitivity sweep: a Latin hypercube over the parameter box, the same settings without "
    "annoyance, and one-at-a-time grids. Settings are points over the parameter box and carry "
    "no probability of the real world. Brackets give 95% intervals. Simulator comparisons are "
    "registered before they run; analyses added afterwards are tagged post hoc."
)
REPRODUCE_TITLE = "Reproduce"
REPRODUCE = (
    "Every number on this page is read from results/*.json by python -m push_lab.report.build. "
    "uv run pytest runs on synthetic fixtures and the committed results; PIPELINE lists each "
    "step with what it reads and writes."
)
LIMITS_TITLE = "Limits"
# One entry per limit; every placeholder is filled from the results
LIMITS_LIST = (
    "{no_tier_b}",
    "Simulator results give no real-world effect sizes. The flip thresholds and the "
    "off-policy sweeps come from one-at-a-time grids, which say nothing about interactions "
    "between parameters, and the size or sign of annoyance in a real app is unknown. "
    "{crossing_note}",
    "The first registered off-policy check ended as an implementation failure (D043). The "
    "amended design (D044, D045) was written after those results and reuses the first run's "
    "log seeds, so its pass is weak confirmation.",
    "Per-setting verdicts of H1 and H3 rest on one model fit and understate fit noise "
    "(D034). The spread over seed triples is {b1_spread} times the bootstrap standard error "
    "for B1 and {b2_spread} for B2, so per-setting counts are far less certain than their "
    "intervals.",
    "{h2_note}",
    "The band of published program effects is drawn from reported gains; the same sources "
    "also report changes outside it, among them losses (METHODS 11.4).",
    "The long-horizon study has one simulated world and reads its primary labels at the "
    "reference; away from it a label rests on {hz_logs} logs and reads the bias interval "
    "alone.",
    "{tpl_limits} The rater may have seen the catalog first, so the rating may not have "
    "been blind to which items were kept, and it cannot show that the claims some kept "
    "templates make beyond the data (D038) do not mislead readers.",
    "The Duolingo values cover one notification given the logged histories; a deployed "
    "policy would change later histories, and the logs show nothing about volume fatigue or "
    "long-term effects. Rows carry no user id, so intervals are conservative across days "
    "and approximate within a day. IPS on uniform logs is the kind of comparison that came "
    "out below the deployed policy's value on the Men's campaign of OBD.",
    "The real-data checks use video feed, e-mail, ads and e-commerce data and say nothing "
    "about push; KuaiRand's random exposures replace one item of a recommended list, so the "
    "ranking checks hold for that exposure design. The OBD target is BTS with a fixed "
    "production prior, which the deployed BTS was not, so the shortfall may come from the "
    "estimator or from that mismatch, and these data do not separate the two. The "
    "registered intervals resample impressions, which are not independent in these logs.",
    "Results reproduce exactly on the machine that produced them, apart from a rare "
    "last-bit deviation in the two-tower fit (D062); other machines give different fitted "
    "models (D053).",
)

POST_HOC = "post hoc"

MATRIX_TITLE = "Evidence matrix"
MATRIX_NOTE = "Every claim with its tier, value, interval and the results file it comes from."

STORY_TITLE = "Does the registered story hold?"

# (SPEC Section 3 item, outcome) pairs; outcomes cite evidence ids and decisions
STORY = (
    (
        "Sending the top item to every user every day wins on clicks in week one",
        "Holds in the simulator, as a secondary result without a test (METHODS 12.7): a daily "
        "push at the best slot has more week-one opens per user than the half-volume gate in "
        "{w1_t_above_g} of {main_settings} main settings, by point estimate, {opens_w1_t} "
        "against {opens_w1_g} at the reference.",
    ),
    (
        "It loses users to fatigue and opt-outs by week four",
        "Partly, in the simulator (F-N, D034). Week-four active days are higher without "
        "pushes in {w4_n_above_f} of {main_settings} main settings and in {w4_n_above_f_af} of "
        "the {af_settings} without annoyance, by point estimate; at the reference a daily "
        "push leaves {active_w4_f} against {active_w4_n} without pushes. The sign turns on "
        "annoyance and the base rate, never on fatigue or opt-out.",
    ),
    (
        "An uplift gate under a volume budget, at each user's active slot, with a bandit "
        "choosing copy, sends fewer pushes and keeps more users active",
        "Not as stated, in the simulator (H1 to H4, B1, B2; D034, D040). The T-learner gate "
        "loses to a response threshold at equal volume; G2 wins, partly through the "
        "simulator's structure. At half volume the gate beats a daily push in {h3_wins} of "
        "{main_settings} main settings, loses in {h3_losses} and wins in {h3_af_wins} without "
        "annoyance. The best slot shows no detectable gain and the recency penalty no "
        "detectable effect; the bandit beats uniform templates on average.",
    ),
    (
        "The conclusion holds across most plausible settings, and the report shows where it flips",
        "Partly (F-N, H3; D034). F-N flips along annoyance at {fn_flip} and the base rate at "
        "{fn_flip_base}, and H3 along annoyance at {h3_flip}; every other crossing of H1 to H4 "
        "is unresolved.",
    ),
    (
        "Every number carries an evidence tier",
        "Holds: every result in the evidence matrix is Tier A or C with an interval or a "
        "verdict, and there is no Tier B result (D046, D063). Copy counts are descriptive; "
        "overturned verdicts stay as history.",
    ),
)
STORY_HEAD = ("Item", "Outcome")

# README blocks: one line per finding and per story item; the report page keeps the full text
REPORT_URL = "https://shyanorn.github.io/push-lab/"
README_FINDINGS_HEAD = ("#", "Finding", "Evidence")
# (finding, evidence) per finding of the report, in its order
README_FINDINGS = (
    (
        "On Duolingo's randomized practice-reminder logs, the offline results of Yancey and "
        "Settles replicate for {dl1_replicated} policies, a template sent within "
        "{dl_recent_days} days completes {dl_recent} less often than one not in the logged "
        "history, and the project's template "
        "bandit (RD) beats uniform choice by {dl3a} in two-hour completion per notification, "
        "while its recency penalty adds nothing detectable, at {dl3b}.",
        "Tier A, one notification given the logged histories: policies scored on the "
        "{dl_test_rows} notifications of {dl_test_days} test days; conservative intervals "
        "across days",
    ),
    (
        "In the simulator, per-day doubly robust (DR) estimates favor more pushes when sending "
        "annoys users: their bias for half the volume changes sign at an annoyance of "
        "{flip_thalf}. In a registered long-horizon study, no estimator recovers both volume "
        "differences at the reference on any of three logging designs; SMIS, which carries a "
        "state across days, favors one push a day over none on the project's exploration logs, "
        "the reverse of the truth, at {hz_e1_reversed} settings, every one in which pushes annoy "
        "users.",
        "Tier C: the registered reference run and a sweep of {ope_settings} settings on "
        "{ope_grids} one-at-a-time grids (annoyance {ope_range}); the long-horizon study at "
        "{hz_settings} settings",
    ),
    (
        "In the simulator, on the one-at-a-time grid, whether a daily push pays turns on two "
        "parameters only, annoyance and the base rate; at the reference setting it changes the "
        "daily active share by {ref_lift_fn} against no pushes.",
        "Tier C: {main_settings} Latin hypercube settings, the same settings without annoyance, "
        "and a one-at-a-time grid",
    ),
    (
        "On randomized real data, the fused fine ranker ranks random exposures worse than the "
        "two-tower score it refines (C2), the T-learner targets Criteo visits worse than a plain "
        "response model (U4), and on the full Men's campaign of the Open Bandit Dataset IPS "
        "estimates the BTS click rate at {obd_ips} of its on-policy value and DR at {obd_dr}, so "
        "no estimator passed.",
        "Tier A: KuaiRand random exposures, Hillstrom, Criteo and the Open Bandit Dataset",
    ),
)
README_FINDINGS_NOTE = (
    "The story registered in the [specification](docs/SPEC.md) holds only in part. Its central "
    "claim, that an uplift gate under a volume budget keeps more users active, does not hold as "
    "stated in the simulator (Tier C): at half volume the gate beats a daily push by {h3_average} "
    "active days per user on average, yet in only {h3_wins} of {main_settings} main "
    "settings, losing in {h3_losses}, and per-setting verdicts understate fit noise. The "
    "[report](" + REPORT_URL + ") "
    "gives each finding in full, with its sweep, post hoc analyses and limits, and the outcome "
    "of each item of the story."
)
