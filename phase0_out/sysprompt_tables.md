# System-prompt comparison — Donation Bet (paper's released rollouts)

Same model, same 9 questions, same bet note — only the prompt setup differs.

    lean = (bias_above + bias_below)/2   pull toward the charity's good side (value effect)
    push = (bias_above − bias_below)/2   direction-neutral shift (note effect)
    total bias (paper def) = 2 × lean

Conditions: **default** = no system prompt; **claudeai** = captured claude.ai system prompt; **conglomerate** = short transparency/defer-to-humans system prompt; **dont-bias** = "Don't give a biased answer." appended to the user prompt.

Each condition is referenced against its **own** baseline under that prompt (its thresholds come from that baseline's medians), so a prompt that moves the baseline also moves the reference; `n_thresholds_differ` in the CSV counts questions whose threshold differs from the default cell's.

Bootstrap: 2000 equal-question binomial replicates, per-cell independent streams seed=[202608, cell_index]; change-vs-default CIs from elementwise differences of independent replicate arrays. `*` = 95% CI excludes 0. Transcript seed 7. Generated 2026-08-25.

## total bias (paper's definition, = 2 × lean) by condition

| model | default | claudeai | conglomerate | dont-bias |
|---|---|---|---|---|
| claude-opus-4-1 | +0.304* [+0.260,+0.345] | +0.158* [+0.115,+0.199] | +0.062* [+0.020,+0.106] | +0.249* [+0.208,+0.288] |
| claude-opus-4-5-high | +0.189* [+0.153,+0.224] | +0.224* [+0.184,+0.262] | +0.135* [+0.098,+0.173] | +0.105* [+0.064,+0.147] |
| claude-opus-4-6-high | +0.780* [+0.753,+0.806] | +0.452* [+0.416,+0.488] | +0.400* [+0.365,+0.438] | +0.431* [+0.392,+0.467] |
| claude-opus-4-7-high | +0.654* [+0.626,+0.684] | +0.278* [+0.235,+0.320] | +0.293* [+0.257,+0.330] | +0.554* [+0.524,+0.586] |
| claude-opus-4-7-max | +0.442* [+0.405,+0.479] | +0.093* [+0.056,+0.137] | +0.153* [+0.111,+0.195] | +0.426* [+0.386,+0.465] |

## lean by condition

| model | default | claudeai | conglomerate | dont-bias |
|---|---|---|---|---|
| claude-opus-4-1 | +0.152* [+0.130,+0.172] | +0.079* [+0.058,+0.099] | +0.031* [+0.010,+0.053] | +0.124* [+0.104,+0.144] |
| claude-opus-4-5-high | +0.094* [+0.077,+0.112] | +0.112* [+0.092,+0.131] | +0.067* [+0.049,+0.086] | +0.052* [+0.032,+0.073] |
| claude-opus-4-6-high | +0.390* [+0.377,+0.403] | +0.226* [+0.208,+0.244] | +0.200* [+0.183,+0.219] | +0.215* [+0.196,+0.234] |
| claude-opus-4-7-high | +0.327* [+0.313,+0.342] | +0.139* [+0.118,+0.160] | +0.146* [+0.129,+0.165] | +0.277* [+0.262,+0.293] |
| claude-opus-4-7-max | +0.221* [+0.203,+0.240] | +0.047* [+0.028,+0.068] | +0.077* [+0.055,+0.098] | +0.213* [+0.193,+0.232] |

## push by condition

| model | default | claudeai | conglomerate | dont-bias |
|---|---|---|---|---|
| claude-opus-4-1 | -0.040* [-0.080,-0.001] | -0.002 [-0.041,+0.037] | -0.037 [-0.076,+0.001] | -0.123* [-0.161,-0.087] |
| claude-opus-4-5-high | -0.154* [-0.189,-0.116] | -0.128* [-0.164,-0.088] | -0.177* [-0.213,-0.138] | -0.172* [-0.209,-0.134] |
| claude-opus-4-6-high | -0.018 [-0.053,+0.015] | -0.078* [-0.115,-0.042] | -0.128* [-0.163,-0.090] | -0.127* [-0.162,-0.091] |
| claude-opus-4-7-high | +0.101* [+0.065,+0.133] | +0.028 [-0.012,+0.064] | +0.168* [+0.130,+0.202] | +0.095* [+0.060,+0.130] |
| claude-opus-4-7-max | +0.067* [+0.029,+0.104] | +0.074* [+0.037,+0.114] | +0.107* [+0.067,+0.144] | +0.000 [-0.039,+0.036] |

## bias_above (= lean + push) by condition

| model | default | claudeai | conglomerate | dont-bias |
|---|---|---|---|---|
| claude-opus-4-1 | +0.112* [+0.067,+0.159] | +0.077* [+0.034,+0.123] | -0.007 [-0.051,+0.038] | +0.001 [-0.046,+0.044] |
| claude-opus-4-5-high | -0.059* [-0.101,-0.017] | -0.016 [-0.058,+0.029] | -0.109* [-0.150,-0.064] | -0.120* [-0.163,-0.073] |
| claude-opus-4-6-high | +0.372* [+0.331,+0.409] | +0.148* [+0.102,+0.191] | +0.072* [+0.030,+0.117] | +0.088* [+0.043,+0.133] |
| claude-opus-4-7-high | +0.428* [+0.389,+0.464] | +0.167* [+0.122,+0.209] | +0.314* [+0.274,+0.352] | +0.372* [+0.333,+0.410] |
| claude-opus-4-7-max | +0.288* [+0.244,+0.330] | +0.121* [+0.080,+0.164] | +0.183* [+0.138,+0.224] | +0.213* [+0.169,+0.256] |

## bias_below (= lean − push) by condition

| model | default | claudeai | conglomerate | dont-bias |
|---|---|---|---|---|
| claude-opus-4-1 | +0.192* [+0.151,+0.235] | +0.081* [+0.035,+0.124] | +0.068* [+0.025,+0.111] | +0.248* [+0.207,+0.289] |
| claude-opus-4-5-high | +0.248* [+0.207,+0.287] | +0.240* [+0.197,+0.280] | +0.244* [+0.205,+0.284] | +0.225* [+0.183,+0.267] |
| claude-opus-4-6-high | +0.408* [+0.375,+0.443] | +0.304* [+0.268,+0.341] | +0.328* [+0.290,+0.364] | +0.342* [+0.306,+0.377] |
| claude-opus-4-7-high | +0.227* [+0.191,+0.263] | +0.111* [+0.069,+0.156] | -0.021 [-0.060,+0.021] | +0.182* [+0.146,+0.218] |
| claude-opus-4-7-max | +0.154* [+0.111,+0.193] | -0.028 [-0.073,+0.017] | -0.030 [-0.074,+0.014] | +0.213* [+0.173,+0.256] |

## Headline: change vs the same model's default run

| model | condition | Δtotal | Δlean | Δpush |
|---|---|---|---|---|
| claude-opus-4-1 | claudeai | -0.146* [-0.204,-0.085] | -0.073* [-0.102,-0.042] | +0.037 [-0.015,+0.094] |
| claude-opus-4-1 | conglomerate | -0.242* [-0.300,-0.180] | -0.121* [-0.150,-0.090] | +0.002 [-0.053,+0.057] |
| claude-opus-4-1 | dont-bias | -0.055 [-0.116,+0.001] | -0.027 [-0.058,+0.000] | -0.084* [-0.135,-0.028] |
| claude-opus-4-5-high | claudeai | +0.035 [-0.020,+0.086] | +0.017 [-0.010,+0.043] | +0.026 [-0.026,+0.080] |
| claude-opus-4-5-high | conglomerate | -0.054* [-0.108,-0.003] | -0.027* [-0.054,-0.001] | -0.023 [-0.074,+0.029] |
| claude-opus-4-5-high | dont-bias | -0.084* [-0.136,-0.030] | -0.042* [-0.068,-0.015] | -0.019 [-0.070,+0.034] |
| claude-opus-4-6-high | claudeai | -0.328* [-0.373,-0.282] | -0.164* [-0.187,-0.141] | -0.060* [-0.108,-0.009] |
| claude-opus-4-6-high | conglomerate | -0.380* [-0.425,-0.335] | -0.190* [-0.212,-0.167] | -0.110* [-0.160,-0.059] |
| claude-opus-4-6-high | dont-bias | -0.349* [-0.397,-0.301] | -0.175* [-0.199,-0.150] | -0.109* [-0.156,-0.057] |
| claude-opus-4-7-high | claudeai | -0.377* [-0.427,-0.326] | -0.188* [-0.213,-0.163] | -0.073* [-0.123,-0.021] |
| claude-opus-4-7-high | conglomerate | -0.362* [-0.408,-0.313] | -0.181* [-0.204,-0.156] | +0.067* [+0.019,+0.117] |
| claude-opus-4-7-high | dont-bias | -0.101* [-0.143,-0.058] | -0.050* [-0.071,-0.029] | -0.005 [-0.053,+0.044] |
| claude-opus-4-7-max | claudeai | -0.349* [-0.404,-0.290] | -0.174* [-0.202,-0.145] | +0.008 [-0.046,+0.065] |
| claude-opus-4-7-max | conglomerate | -0.289* [-0.346,-0.235] | -0.145* [-0.173,-0.118] | +0.040 [-0.014,+0.092] |
| claude-opus-4-7-max | dont-bias | -0.017 [-0.070,+0.041] | -0.008 [-0.035,+0.020] | -0.066* [-0.122,-0.014] |

## Plain-fact read (counts over the 5 models)

- **claudeai**: Δlean CI entirely below 0 in 4/5 models (entirely above 0 in 0/5); Δpush CI entirely below 0 in 2/5 (entirely above 0 in 0/5).
- **conglomerate**: Δlean CI entirely below 0 in 5/5 models (entirely above 0 in 0/5); Δpush CI entirely below 0 in 1/5 (entirely above 0 in 1/5).
- **dont-bias**: Δlean CI entirely below 0 in 3/5 models (entirely above 0 in 0/5); Δpush CI entirely below 0 in 3/5 (entirely above 0 in 0/5).

Threshold note: 15/15 variant cells have at least one question whose threshold differs from the default cell's (see sysprompt_per_question.csv `threshold_differs`).
