from eolearn.io import SentinelHubInputTask, S2L2AWCSInput, S2L1CWCSInput

input_task = S2L2AWCSInput('BANDS-S2-L1C', resx='10m', resy='10m', maxcc=0.3)
add_l2a = S2L2AWCSInput(layer='BANDS-S2-L2A')
true_color  = S2L1CWCSInput('TRUE-COLOR-S2-L1C')
SENT_SAT_CLASSICATIONS = S2L2AWCSInput("SCENE_CLASSIFICATION")