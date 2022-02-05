import matplotlib.pyplot as plt
import tensorboard as tb
from packaging import version
import grpc

import numpy as np
import pandas as pd

# Validation Runs.

enets_fp = {
    'label': 'EfficientNets (FP32)',
    # 'color': 'b',
    0: 'BRj9fv5PR0yAWkD4z0p5FQ',
    1: 'QRMPo8cVQRqk01JbKZOMjw',
    2: 'DZXKGFneSoW8rj5qZZz3LQ',
    3: 'dD3zay4XTYm6ltpNTGocDg',
    4: '4VwTvygFQ2WFG74GlF8Tqw',
}

enets_int8 = {
    'label': 'EfficientNets INT8 (0-4)',
    # 'color': 'g',
    4652008: 'L2wx6i0dRly0LTG3sA9qpg',
    5416680: 'oXPvlPrSQkKyZlivUrby7w',
    6092072: 'KMC8ULhbQviDC5LN1aj4dA',
    8197096: '7hgKgc31QXm7rVOX8hm0rg',
    13006568: 'G6PJWXMRQyiAMZCEyVfVsA',
    'params': 8,
}


# enets_int8_t2 = {
#     4652008: 'NAxjD6ydQDq25NdJ7RQauA',
#     5416680: 'b1kq2yDsRrSoQORKAoiwHQ',
#     6092072: 'PpcD8wrzRga4S5Qg3RxKfg',
#     8197096: 'EyXpX0NbTnqvQt88F9JdSg',
#     13006568: 'ZjKyJ1nlSqqtGivBytyVdg',
#     'params': 8,
#     'label': 'EfficientNet-Lites (INT8)',
# }


# Parametric Quant

enet0_lsq = {
    'label': 'LSQ',
    # 'color': 'g',
    2: 'mYPAx8Z5QXmyeTx3prbdfQ',
    3: 's8rGb5dCQBWPnD1QyCVZsQ',
    4: 'wY16VBQXSpOcj7mClBMjfA',
    5: 'SGwFwwYaRqSXov6vgGwV9Q',
    6: 'xCru2mpURnOqzAsYRWvrdA',
    7: 'dEG44pIzRQmAA83QnCv52w',
    8: 'gS1RSoyBRfOGm4pGuBJ8BQ',
}


# Non-Parametric Quant

enet0_dynamic_init_max = {
    'label': 'Dynamic Init Max',
    'color': 'b',
    # 2:'IvvJGLjvS7SPDJsLC0jFLA',
    # 3:'pvKoZRvnSG6qfEszYE9JpQ',
    # 4:'rja3bfROSNCAR3rwM2QJBQ',
    # 5:'gwa1KzUVRLKb2ux6hQXHjQ',
    # 6:'RG9CyhLTQQWlpvGO0AqsLg',
    # 7:'YkYjgmu3TVajPWNvRohllg',
    # 8:'80CzYZCOQkOHgFNoRqFAkw',

    2: 'OmSmfq59S0aQYMmD7SYFjg',
    3: 'YKZxhW5ETg6M7F6EDO06bg',
    4: 'waHr3rS9R52JeWCbz1Lntw',
    5: '3fXh0YayQLuZzfjAWKN1kQ',
    6: 'wvkqUkAmTNunCBCdLaq6qw',
    7: 'agYwfLAgRXCa692n5t5R4A',
    8: 'rijJczuQQrWpxiz7EhVMdQ',
}


enet0_dynamic_init_double_mean = {
    'label': 'Dynamic Init Double Mean',
    'color': 'g',
    2: 'HA3kLdFAQReV8IRFwOLnig',
    3: 'aVJ26QKSRLqujYJ2uwlnBQ',
    4: 'FAfkPLlXRLKpfX72gqFWSg',
    5: 'ADvNNAYvSpmyXqsvv3g3hg',
    6: 'Ul3KxPh5TXSamryS5woj0A',
    7: 'nZVltBcqRPuXKJSeKDt0Ow',
    8: 'ZmwIrSA2SvaLhv85BtJwEw',
}

enet0_dynamic_init_gaussian = {
    'label': 'Dynamic Init Gaussian',
    'color': 'r',
    2: 'qS66vBsZQIqXnSjiGabI7g',
    3: 'YtVzqbnzSKmmdKz0pHof2Q',
    4: 'PCogVoW9TwO6WFGeVX31kg',
    5: 'ePpTK8YzQPG6kNFPq8MYrQ',
    6: 'V9yGsRBXQ0ePDXhqjzMJXw',
    7: 'dpHy0z8qS6aL5XED2CdEmQ',
    8: 've9KwJfYRxOiJ5elyNZyOQ',
}


enet0_static_init_max = {
    'label': 'Static Init Max',
    'color': 'c',
    2: '6Wd3QTwfRrahD5wHGV0yCw',
    3: 'goSVltnRRQyJU8QHyJnKYw',
    4: 'nLZUk2fpQ3qgzBGCJgksOg',
    5: '36lJJc0LRCayqMTAXROPiw',
    6: 'sjrggkHXSAGqBFzzsjHaPw',
    7: 'bxM6szNMSra2X4XkLkewNg',
    8: 'YTVldyMXR32Rj9eBXgqaXw',
}


enet0_static_init_double_mean = {
    'label': 'Static Init Double Mean',
    'color': 'm',
    2: 'kJwP08xMQsK1wrQJrxUE9A',
    3: 'cYvs4lQzR9mEyM8sP3szGQ',
    4: 'kJpmA5fyTMCeK6kTkpJoAw',
    5: 'NpC0BZnpRtyfoJnCOW6gdQ',
    6: '1e8dB7GpR62qD4gPat0mfg',
    7: 'uyRv8qxqSWeOI9GlkiW2Zw',
    8: 'PGYSZAO6TVKDaaMkRWjs3A',
}

enet0_static_init_gaussian = {
    'label': 'Static Init Gaussian',
    'color': 'k',
    2: 'm3uqrpJvQfOeZ1Vo8DM9SQ',
    3: 'mtjXBQOeQha1f2Al842B1w',
    4: 'NujiWt8nS6SdePJHK29jQw',
    5: 'gk2ERBL0ReKFaqDIkIMB6Q',
    6: 'c0co5awbRaWIZV9zc4f1Ag',
    7: 'yFLNoPx2Q4yhqnrpWQaY0A',
    8: '0tlDzZrqQC6KNXADNYw4rQ',
}


enet0_static_init_max_surrogate = {
    'label': 'Static Init Max Surrogate',
    'color': 'c',
    'linestyle': 'dashed',
    2: '5J0nHgB6RdK8Fjq7iO9bHQ',
    3: 'gdnekWMDQBqn7V0AaX8nUA',
    4: 'iqD4VndTQOC1Uz1pF7SwTw',
    5: 'eFPyu49dRAq9BaYJ3e15xQ',
    6: 'K4R29vTvQwy6lfnW1IzEPA',
    7: 'Z9lQvYF4TJmQ31UQH5OJ6A',
    8: '6nHX19QbQsKszqNmalGjCg',
}

enet0_dynamic_init_max_surrogate = {
    'label': 'Dynamic Init Max Surrogate',
    'color': 'b',
    'linestyle': 'dashed',
    2: '6VznjGRiQvegtlVIqYG1jA',
    3: 'YeQa1zJwQ7KavXL4nGaCpA',
    4: 'FnP7qU9BQta2OdRlT68M2A',
    5: '3hPRiW3jRo6iLKTjoaqTQQ',
    6: 'JNqKiV5WT92eOjrE2GE9Mg',
    7: 'SpatG4eBRdS2op9rdnN9vA',
    8: 's3fQeZZ8RZ6btvGhyRjjPg',
}


# Cosine Double Mean Dynamic


enet0_dynamic_init_double_cos = {
    'label': 'Base',
    'color': 'b',
    2: 'KlrLYaGKSLqOQhYQcJQdyg',
    3: 'GHg5TYWXSDWBQnaTXh5yZw',
    4: 'meOvRddLRVmVLRTNxK6VMA',
    5: 'S7Add0bUR1yW5daWU4BQuQ',
    6: 'HK1dXQyrQOiMEuJt0IbR4A',
    7: 'r0rUmKvBSXGSduayO0p2tw',
    8: 'yjztXeepRjS01h7MBLhXNg',
}


enet0_dynamic_init_double_cos_sur = {
    'label': 'Surrogate',
    'color': 'm',
    2: 'WfgRIKlcQSGyBNnAY4k6ow',
    3: 'z4LowapiQje4SXSmMPXJ5g',
    4: 'KfnEL9xnRrKYMhg5gQ4aNQ',
    5: 'QLBxcA2lQPWSJrEVqeeVSw',
    6: '6w5HwOXlT42DPtmOeQEwEA',
    7: 'mK4JHaK7QhyRR8vbI81azw',
    8: '1uwvSE5UQXOOUbsDXi7m2Q',
}

enet0_dynamic_init_double_cos_psg = {
    'label': 'Fake PSQ',
    'color': 'g',
    2: '6LzRQpniQ2Gaj8vJnHbfJA',
    3: 'LmSvzYLKTmquk1YurDeljw',
    4: 'azSo4fKZQA2MHWDpsK3Z8w',
    5: 'lhmcp47kR6GHrwm6oRFoaw',
    6: 'K40IdAk2REGnsVAkzY3JrQ',
    7: 'GeZhDnZoSAi0ZdovhrL5IA',
    8: 'yM6pybEsQse6xLPz8Zd1hg',
}

enet0_dynamic_init_double_cos_ewgs = {
    'label': 'Static EWGS',
    'color': 'r',
    2: 'hZrloyqPTJKg6Y1xDhVBfQ',
    3: 'psYA2BOFQ2mx17Zzwww28g',
    4: 'Z1BoDnvKRMKpzbyDHeoxrA',
    5: '5v62erxOQ5qAvkVyyw8Eig',
    6: 'cg3B817URDidCj42IVXEmg',
    7: '8H0Zx9r7QqG9FRFAp31HyQ',
    8: 'DpHROXlLS8agaADI4xl8xw',
}


#
# New Sweeps after dynamic correction
#


enets_fp32_t2 = {
    4652008: 'oIBswEoqTme2oLlkkhx4Xw',
    5416680: 'lHDJHMe2RamzzWwF4DF1hQ',
    6092072: 'RiESL4q5QGuzCO66fYZ70Q',
    8197096: 'LvwDTyJbRwKlvE77mBtWhw',
    13006568: 'rdXGZbqPTyuOem2n2s7K0Q',
    'params': 32,
    'label': 'EfficientNet-Lites (FP32)',
}


# enets_int8_t2 = {
#     4652008: 'NAxjD6ydQDq25NdJ7RQauA',
#     5416680: 'b1kq2yDsRrSoQORKAoiwHQ',
#     6092072: 'PpcD8wrzRga4S5Qg3RxKfg',
#     8197096: 'EyXpX0NbTnqvQt88F9JdSg',
#     13006568: 'ZjKyJ1nlSqqtGivBytyVdg',
#     'params': 8,
#     'label': 'EfficientNet-Lites (INT8)',
# }


enet0_dynamic_init_double_mean_t2 = {
    2: 'GHq9K5XuQNyxqD6GGlcPQg',
    3: 'brXMuvyRS1W22FR24KaCXA',
    4: '8gGvrbG6RC2YqiRfQzXANw',
    5: 'v6hkFRfbQA6Tbw3aTPRmAA',
    6: 'DvKtwOR6Tl6ONeN879JQbQ',
    7: 'mPkKzrxrR2yWf9koRDCaUw',
    8: '21cNoDLSQ9GTwvSw0da43g',
    'params': 4652008,
    'label': 'EfficientNet-Lite0 Dynamic Quant Init Double Mean',
}

enet0_static_init_double_mean_t2 = {
    2: 'J4NdnC9XRzOQnkymDBAjxA',
    3: 'Xo9GWJ1BQbqjF6x3WJjvMg',
    4: 'f9clPhE3QHeAJH8nvrSJqQ',
    5: 'LVXphdloRXSR0FmhZLwDxg',
    6: 'asOKaXvkTzKBuQhk89E0CQ',
    7: 'xqQR8WZsQBGJTGXc42qgtA',
    8: 'bG0YRHwQSOKpYyNroDiYRA',
    'params': 4652008,
    'label': 'EfficientNet-Lite0 Static Quant Init Double Mean',
}

enet0_dynamic_init_gaussian_t2 = {
    2: 'NbEovMSJRpmWQUN7kxAN9Q',
    3: '0VHp8egmSXGcEMrJwHJUrQ',
    4: 'RJldVC0UQim8fYoJJj9rUg',
    5: 'OCj3BknTRyWvAGVT6d2PLA',
    6: 'mZkrNxWRSo6jazfdJQjxOg',
    7: 'pP5neKFJRa6LQWYeGigR2w',
    8: 'rXW4RcMxQpOE64YKknZsDQ',
    'params': 4652008,
    'label': 'EfficientNet-Lite0 Dynamic Quant Init Gaussian',
}

enet0_static_init_gaussian_t2 = {
    2: 'XnQ2t7qxR9yFrLtlkbWONQ',
    3: 'Sv5NEXHGQ5qG21llbjUa0g',
    4: 'cWEEkRk2T2uxEsRtdPUPyA',
    5: '8sY3PAc7Qo6WMKaWg4qzyw',
    6: '6RkmMxdeRGO6BPED6jljAw',
    7: 'Vslt1nD5S9i0eCz18rOuVA',
    8: '4xIj338JT92W5kxZ9f4MGQ',
    'params': 4652008,
    'label': 'EfficientNet-Lite0 Static Quant Init Gaussian',
}

enet0_mixed = {
    0: 's8XTz88OSw6Fg8YJtq2yog',
    'label': 'EfficientNet-Lite0 Mixed Precision',
}


enet0_lr_best = {
    2: '',
    3: '9nwVcMdPRRGJtjDjrZpCeg',
    4: 'kX80UjhnQxWdwEdzQDTNzQ',
    5: 'KMwHB3liQM2WR7jSzE7nNQ',
    6: 'mx3Gs5p4T0S9KSBXeFqAvg',
    7: 'yZQjrnLBS9iG3DFeUzoaPg',
    8: 'vq1yDQlLSgSbNH0he3QE5A',
    'params': 4652008,
    'label': 'EfficientNet-Lite0 (3-8 Bits)',
}

enet0_dynamic_lsq = {
    2: '5QmhaVjKQJ21PuQndhjS5w',
    3: 'acnxrPWPQUe9sMoWDnar9w',
    4: 'm9nfdsZUQNad0k577p3x2Q',
    5: 'CeTVs6kyQki18J6HyaF06g',
    6: 'qdXDLDiwTaGrg4EsEtjBQA',
    7: 'I01f3bJpR62asyFksRrX6Q',
    8: '9wqXBWQKR8qfpyQwpF8z3Q',
    'params': 4652008,
    'label': 'EfficientNet-Lite0 LSQ',
}


enet0_mixed_8 = {
    0.5: 'BfqJm7rsRYSPm9Igwm9VRg',
    0.75: 'I3RVK4FXRby7YF1gytxSOw',
    1.0: 'hjFmom8pTwOrZhwwauZPZg',
    1.25: '2NHLJfzxQyS3ugxfAhOOHg',
    1.5: 'XLwyYn0rTOyhafpy0vYUDg',
    1.75: 'dNU3243wRnOnjcXyGk2XNg',
    2.0: '4gRxTD4JSZWDYX44ZYM1ww',
    2.25: 'b2Y0DDiDRuKRA3PUzT2N9Q',
    2.5: 'ewbosqbfRTO13BtxAxwjgQ',
    2.75: 'd5d0lJJUQZups43m35tBHQ',
    3.0: 't9OXcxOPT4yi3WqLdRKLCw',
    3.25: 'Mcg57HqZRiywmfnzqXb8qQ',
    3.5: 'tQEtg1enT1OPjogd7FBLsw',
    3.75: '3ZGkHm0nS2aBm54KawqkAA',
    4.0: 'jDEpeS66RhqGJaD73At8qQ',
    4.25: 'gRhrXPWSRWy3R6u6ZwhXKw',
    4.5: 'FENZ1HNmQfyZYJHzgXUq8g',
    4.75: 'ywRdvzYQR2WTeKyvdRuXrA',
    5.0: 'hJyU7YsoRqKjrJf2JD34lQ',
    # 'params': 'mixed',
    'label': 'EfficientNet Mixed (Start 8bit)',
}


enet0_mixed_4 = {
    1.0: 'vOXeFEPKQlOu71Imk6NIbg',
    1.25: 'Ue7qJd3kRWGAF2l2HWlOEw',
    1.5: 'wv32wt6vTDmjcGaWzXy9Yw',
    1.75: 'Sxa2Bqn3SGK4vlK4xQ14TA',
    2.0: 'vHwxbPKmR8GMtYFRzGt7uQ',
    2.25: 'C2x3rAM5TeqALJMQ90BpZA',
    2.5: '5uBFXcQWQAaSEl71fQDNEA',
    2.75: 'uNg6HGxjS9m1FxNqK5CX2w',
    3.0: 'UgivLu3bSl2l6EydK3lXPQ',
    3.25: 'tJoAavSRRIyWvOKJgEwDdg',
    3.5: 'zr2NDALbRa2kRACMTBTEwA',
    3.75: 'WsKpFjvzSMep8jORSRwPLQ',
    4.0: 'bvONnY97QxO1bVQluAKt3w',
    4.25: 'cM62vormRIaL7OdmiwqKsg',
    4.5: 'Lnl8YwyVTWaas8JmeVKMzw',
    4.75: '6f3DvWziQEaygRfl6p8RBA',
    5.0: 'QbdFAh2ZSlSYJXnvK65Rbg',
    # 'params': 'mixed',
    'label': 'EfficientNet Mixed (Start 4bit)',
}

enet0_static_lr = {
    '2_0.0001': 'jWoecMWnT1eBDLljEJa2kw',
    '2_0.001': 'D1wSdKMwRmGvL6Xq5ckycw',
    '2_0.01': 'ulbc5YVzQKGIphKkL82SDA',
    '2_0.1': 'ANOi0dHOSJSJyqcNROg73A',
    '3_0.0001': 'rKTXZqHGSdGAluOKeSOevQ',
    '3_0.001': 'li5gkK6rQImofqLw7UP5IQ',
    '3_0.01': '9nwVcMdPRRGJtjDjrZpCeg',
    '3_0.1': 'y3f9xvoKQnCaa2om4QBJ5A',
    '4_0.0001': 'tOPfgpDtQg6zi2vfAOQ17w',
    '4_0.001': 'ows8qJCqRnCLDszcZv3vUw',
    '4_0.01': 'kX80UjhnQxWdwEdzQDTNzQ',
    '4_0.1': '6wAtYEI4S3GnOy0MDm1qQw',
    '5_0.0001': 'jCzBs6x0TwSLUB2rTorgDg',
    '5_0.001': '4iXpp7wgTY6LPW13XjdLgQ',
    '5_0.01': 'KMwHB3liQM2WR7jSzE7nNQ',
    '5_0.1': 'MPomRAs9QKOZgXtyWSWpXw',
    '6_0.0001': '6JgDoSdLTAySwTORfQU2SQ',
    '6_0.001': 'uQTvwYikToqD5Oq8xnBBNg',
    '6_0.01': 'mx3Gs5p4T0S9KSBXeFqAvg',
    '6_0.1': 'kjMJrPaeQuW97iBadeXRDQ',
    '7_0.0001': 'T61M9TAmTBqyTlrxsGZE9Q',
    '7_0.001': 'yZQjrnLBS9iG3DFeUzoaPg',
    '7_0.01': 'LGQBdb6qTLCK91Lzjg8O9g',
    '7_0.1': 'bBLW2N4RTzuE3QguzQYa9g',
    '8_0.0001': 'RoRByyMeSZKlmOjqY19PZQ',
    '8_0.001': 'vq1yDQlLSgSbNH0he3QE5A',
    '8_0.01': 'cya49VDMTYSnQbLFEp8CmQ',
    '8_0.1': 'p2VntR8LRFOGiD2lkt0BVQ',
}

enet0_static_init_4b = {
    'max_max': '7I2DKoqsQrqAH2O63D4eig',
    'max_doublemean': '4aRGN3vNSA6o0qA4TaRcPw',
    'max_gaussian': 'U1fqXpfwROG6hpWWiwqQPQ',
    'max_p999': 'DSgmOUWnRYWUnERltqyzfg',
    'max_p9999': 'PnWsb45aSlOA9cdunJg0aw',
    'max_p99999': 'IjXkNHrVRZGyNaQYRIdi4A',
    'max_p999999': 'jrNQkT1LRe6jxd91EzIJgA',

    'doublemean_max': 'ulnMJxo3TveGAtD3ADrpwg',
    'doublemean_doublemean': '15vphyrbSyuLlgL83m9KKg',
    'doublemean_gaussian': 'U6R18149SyagJ54Mo9tEiQ',
    'doublemean_p999': 'ZjF4WtL9RgqPZrA9K3RC0w',
    'doublemean_p9999': 'WLSYbxBZT9adklLBYgqx5w',
    'doublemean_p99999': 'RleVz6NfRECO08GQ2xt3Wg',
    'doublemean_p999999': 'WDHQazB2TOCZaTqKY5orFw',

    'gaussian_max': 'hn06OO2YRtGyQcxtYYjOLw',
    'gaussian_doublemean': 'jDPh2vqwRmGa0Baio2AalA',
    'gaussian_gaussian': 'iLAKzaklSQGurdFMxDtJOg',
    'gaussian_p999': 'YFPdCRt9SvWqmC0A5h12wg',
    'gaussian_p9999': 'Nk7E7qWfRcKWNt434zz66Q',
    'gaussian_p99999': 'QptpKhnOSgCVcevL9By33g',
    'gaussian_p999999': 'jH56RpOCRnSJKYOPnGRJXw',

    'p999_max': 'pDCdoFmRQzSn4dboj4zuEg',
    'p999_doublemean': 'tAt1Ktk4TpaZBPsGa2voPQ',
    'p999_gaussian': 'IVeer3DYTl242ljzJGvC1Q',
    'p999_p999': '6jRFptQ7RxiVYVdJNnDmlw',
    'p999_p9999': 'RyVfWlEqSemvRSXprtLWtg',
    'p999_p99999': 'rkl2oC3RTzSqk1euzW6j5A',
    'p999_p999999': 'ASVLi0vnT7OJdlQRkdnhKQ',

    'p9999_max': 'rcPn5Z82RNaiWk3gYxTReg',
    'p9999_doublemean': 'P9QkR3cDTj2FJSPgHkILBg',
    'p9999_gaussian': 'jxqM61lrQsKWOsMHaEWTWA',
    'p9999_p999': 'wLLLg9dBQNuDc2EEEDIdnA',
    'p9999_p9999': 'lzS0sNldSzyjsePgAmfQpA',
    'p9999_p99999': 'aVX8hplHQcaZ5Qqapkgu3g',
    'p9999_p999999': 'QSaX1VT1RfmxbnHQ3HxiXg',

    'p99999_max': 'kafwv81VRpWgGXgKRxO3AQ',
    'p99999_doublemean': 'WixGrMnBRyG2fWnFG7PSAA',
    'p99999_gaussian': '13nylvJTQB2KN08exyR6rQ',
    'p99999_p999': '7KCWzUyaS8qQa11CQL7J5A',
    'p99999_p9999': 'DnBhxskER4Cjc5gyy8BATw',
    'p99999_p99999': 'I5vAwZtyT92EAPGVXgWBxw',
    'p99999_p999999': 'efQY9aUxQ3GEmxFpH53Sjg',

    'p999999_max': 'ig53dNZMQKy385FJmoWVCA',
    'p999999_doublemean': 'DeBoiOFyQeyaMxJlmjt5wQ',
    'p999999_gaussian': 'd5hxkVuPRRalAjkbwd7azg',
    'p999999_p999': 'iE9LXd1XSEyTqmtbKO8x0A',
    'p999999_p9999': 'g47iG74gRVOXH5WjabdSVw',
    'p999999_p99999': 'Hj8cmCdgRPGrDVvEMvWsWw',
    'p999999_p999999': 'm2szxYF4TVqfDIjQ5OmHyg',
}


enet0_static_init_3b = {
    'max_max': 'uRkyLVXHRI2RnzFyxx2BLg',
    'max_doublemean': 'Ag1XfkXdRwiBvbioaA1NUQ',
    'max_gaussian': 'Adptwcm1SmCbzVJ18Tr43A',
    'max_p999': 'LtOkikgwR6ubngOy5lCLhg',
    'max_p9999': 'FEFmoTN7SJOZGWe1bmLpwQ',
    'max_p99999': 'qiWPD3w8QFmxXvk2c0YjmQ',
    'max_p999999': 'qV22RLlcQeCGN3LW0oKbJA',

    'doublemean_max': '8dKoewxOQV6LlnyymPbbOg',
    'doublemean_doublemean': 'Tvp8kjYURJ6xFpHw5KQvOw',
    'doublemean_gaussian': 'DzDr17udTnivVQlTbBXd1Q',
    'doublemean_p999': '6hWkjL3OTLGcANGoHzKVIA',
    'doublemean_p9999': 'm8nvD0qPSBe3YnWuXOLLyQ',
    'doublemean_p99999': 'pAqAXIm7REu1SWjakNAaNw',
    'doublemean_p999999': 'BmWbYNf9RMaSg6PamiBZkw',

    'gaussian_max': '9044XvMlS96g1ljurWLI7w',
    'gaussian_doublemean': 'Q6RFDe31Qe2xK8eTzsQeCQ',
    'gaussian_gaussian': 'dnWamizRTdS4Qt6Oz7Qk9w',
    'gaussian_p999': '8YYKZvSnSke8ecyAyjwnYg',
    'gaussian_p9999': 'Vbvayi1FQHadjU7DdUGPhA',
    'gaussian_p99999': 'bXL7RwouRLyxBSQWt3wQkA',
    'gaussian_p999999': 'XVY4qKtKRQenWsXLJXqokQ',

    'p999_max': 'vi5i2xJvT16bPnilErd2Pw',
    'p999_doublemean': '6xFdtew1TEGFbjJLS6AoAg',
    'p999_gaussian': 'KKRKyqPaQ7C3a6Vr2Twreg',
    'p999_p999': '14P9POD3SaWcdfDXngv2Zg',
    'p999_p9999': 'FoR5LEqPS7ygbro9OwVvZQ',
    'p999_p99999': 'TVXrZRj4T0qZDOyx3OWLLg',
    'p999_p999999': '3tVBnoxYSdapnd2UX7RowA',

    'p9999_max': 'w6WkIj66TOmtvi93bbJUNA',
    'p9999_doublemean': 'n2IoTlvbR06J68cl5nlpvg',
    'p9999_gaussian': 'idkI2juaRUqqezAFE6Czww',
    'p9999_p999': 'VbinTYLfRB2My65je0AsNg',
    'p9999_p9999': 'v9GMOG6oSyyFveziyWrpQw',
    'p9999_p99999': 'rWenj2tKT7WfVNJn283BBQ',
    'p9999_p999999': 'Mmv7OxytQ6uAxFOveMn7tQ',

    'p99999_max': 'dASV9OpdQZeHxq8hkQpjPw',
    'p99999_doublemean': 'TPvW5jV0S5aNo8eFpD22PQ',
    'p99999_gaussian': 'G6uxYvwGQ6KwQX24BhzUDw',
    'p99999_p999': 'sVpijtdCQHq5akXnpmUQkg',
    'p99999_p9999': 'YoRpeTb9RvqT1ivi9uRrOQ',
    'p99999_p99999': 'HBr4e29dQZaplKDU00FdoQ',
    'p99999_p999999': '9po6ogmlQD6Y7Gbr7mPVwg',

    'p999999_max': 'NvO4bpV0TjeI27K52FaBGQ',
    'p999999_doublemean': 'knbuV66eTyyk6DFLdlK2eA',
    'p999999_gaussian': 'XcMU4VkpRZKoLFPoT5gOgA',
    'p999999_p999': 'neGnvMv6Ra6OmIX4dupQww',
    'p999999_p9999': 'wTD1jzN6S8u4Xd5gW1pnGQ',
    'p999999_p99999': '6NqbGJPfR3mtYC3M8Gsbhw',
    'p999999_p999999': 'BBSZp7sXQHqWiz6EReRLvQ',
}

# Competitor Performance.

competitors = {

    'pact_resnet50': {
        # https://arxiv.org/pdf/1805.06085.pdf
        'eval_err': [1 - 0.722, 1 - 0.753, 1 - 0.765, 1 - 0.767],
        'size_mb': np.array([2, 3, 4, 5]) * 25636712 / 8_000_000,
        'name': 'PACT ResNet50',
        'alpha': 1.,
        # no first and last layer quant
    },

    'pact_resnet18': {
        # https://arxiv.org/pdf/1805.06085.pdf
        'eval_err': [1 - 0.644, 1 - 0.681, 1 - 0.692, 1 - 0.698],
        'size_mb': np.array([2, 3, 4, 5]) * 11679912 / 8_000_000,
        'name': 'PACT ResNet18',
        'alpha': .15,
        # no first and last layer quant
    },

    'pact_mobilev2': {
        # https://arxiv.org/pdf/1811.08886.pdf
        'eval_err': [1 - 0.6139, 1 - 0.6884, 1 - 0.7125],
        'size_mb': np.array([4, 5, 6]) * 3300000 / 8_000_000,
        'name': 'PACT MobileNetV2',
        'alpha': .15,
    },

    'dsq_resnet18': {
        # https://arxiv.org/abs/1908.05033
        'eval_err': [1 - 0.6517, 1 - 0.6866, 1 - 0.6956],
        'size_mb': np.array([2, 3, 4]) * 11679912 / 8_000_000,
        'name': 'DSQ ResNet18',
        'alpha': .15,
    },

    'lsq_resnet18': {
        # https://arxiv.org/abs/1902.08153
        'eval_err': [1 - 0.676, 1 - 0.702, 1 - 0.711, 1 - 0.711],
        'size_mb': np.array([2, 3, 4, 8]) * 11679912 / 8_000_000,
        'name': 'LSQ ResNet18',
        'alpha': .15,
    },

    'lsqp_resnet18': {
        # https://arxiv.org/abs/2004.09576
        'eval_err': [1 - 0.668, 1 - 0.694, 1 - 0.708],
        'size_mb': np.array([2, 3, 4]) * 11679912 / 8_000_000,
        'name': 'LSQ+ ResNet18',
        'alpha': .15,
    },

    'lsqp_enet0': {
        # https://arxiv.org/abs/2004.09576
        'eval_err': [1 - 0.491, 1 - 0.699, 1 - 0.738],
        # number might be incorrect
        'size_mb': np.array([2, 3, 4]) * 5330571 / 8_000_000,
        'name': 'LSQ+ EfficientNet-B0',
        'alpha': 1.,
    },

    'ewgs_resnet18': {
        # https://arxiv.org/abs/2104.00903
        'eval_err': [1 - 0.553, 1 - 0.67, 1 - 0.697, 1 - 0.706],
        'size_mb': np.array([1, 2, 3, 4]) * 11679912 / 8_000_000,
        'name': 'EWGS ResNet18',
        'alpha': .15,
    },

    'ewgs_resnet34': {
        # https://arxiv.org/abs/2104.00903
        'eval_err': [1 - 0.615, 1 - 0.714, 1 - 0.733, 1 - 0.739],
        'size_mb': np.array([1, 2, 3, 4]) * 25557032 / 8_000_000,
        'name': 'EWGS ResNet34',
        'alpha': .15,
    },

    'qil_resnet18': {
        # https://arxiv.org/abs/1808.05779
        'eval_err': [1 - 0.704, 1 - 0.701, 1 - 0.692, 1 - 0.657],
        'size_mb': np.array([5, 4, 3, 1]) * 11679912 / 8_000_000,
        'name': 'QIL ResNet18',
        'alpha': .15,
        # no first and last layer quant
    },

    'qil_resnet34': {
        # https://arxiv.org/abs/1808.05779
        'eval_err': [1 - 0.737, 1 - 0.737, 1 - 0.731, 1 - 0.706],
        'size_mb': np.array([5, 4, 3, 1]) * 25557032 / 8_000_000,
        'name': 'QIL ResNet34',
        'alpha': .15,
        # no first and last layer quant
    },

    # 'hawqv2_squeeze': {
    #     # https://arxiv.org/abs/1911.03852
    #     'eval_err': [1 - 0.6838],
    #     'size_mb': np.array([1.07]),
    #     'name': 'HAWQ-V2 SqueezeNext',
    #     'alpha': 1.,
    # },

    # 'hawqv2_inceptionv3': {
    #     # https://arxiv.org/abs/1911.03852
    #     'eval_err': [1 - 0.7568],
    #     'size_mb': np.array([7.57]),
    #     'name': 'HAWQ-V2 Inception-V3',
    #     'alpha': 1.,
    # },

    'mixed_resnet18': {
        # https://arxiv.org/abs/1905.11452
        'eval_err': [0.2992],
        'size_mb': np.array([5.4]),
        'name': 'Mixed Precision DNNs ResNet18',
        'alpha': .15,
    },

    'mixed_mobilev2': {
        # https://arxiv.org/abs/1905.11452
        'eval_err': [0.3026],
        'size_mb': np.array([1.55]),
        'name': 'Mixed Precision DNNs MobileNetV2',
        'alpha': .15,
    },

    # 'haq_mobilev2': {
    #     # https://arxiv.org/pdf/1811.08886.pdf
    #     'eval_err': [1 - 0.6675, 1 - 0.7090, 1 - 0.7147],
    #     'size_mb': np.array([.95, 1.38, 1.79]),
    #     'name': 'HAQ MobileNetV2',
    #     'alpha': 1.,
    # },

    # 'haq_resnet50': {
    #     # https://arxiv.org/pdf/1811.08886.pdf
    #     'eval_err': [1 - 0.7063, 1 - 0.7530, 1 - 0.7614],
    #     'size_mb': np.array([6.30, 9.22, 12.14]),
    #     'name': 'HAQ ResNet50',
    #     'alpha': 1.,
    # },
}

sur_grads_tb = {"STE" : "YZvB0grsQBi37EVBD7mI1Q",
"Gaussian" : "gNXpasOlQuavXt7UzAlFnw",
"Uniform": "mOKsq40BQX2KDflZYW7t9w",
"PSGD" : "v1KWCVXPSKuxlHW2jF7FjA",
"EWGS" : "DDH4jzTqRluvOFgUEXzYvw",
"Tanh" : "IVFE3f5mSNuPiL0AT44Z0Q",
"InvTanh" : "xsO3b5FoQNKe5w2roo666w",
"Acos" : "jOGi7zWKSEy6R3ZA1bty9A",
}

sur_grads = ["STE,Gaussian,Uniform,PSGD,EWGS,Tanh,InvTanh,Acos",
             "0.65640,0.65800,0.66260,0.65710,0.66550,0.66620,0.67090,0.65770",
             "0.65540,0.65330,0.66290,0.65560,0.66340,0.65850,0.66030,0.65740",
             "0.66110,0.65610,0.66190,0.65410,0.66050,0.65610,0.66150,0.65560",
             "0.65430,0.66830,0.66050,0.65900,0.66570,0.66310,0.66490,0.66100",
             "0.65790,0.66050,0.65840,0.66310,0.66420,0.66460,0.65870,0.65590",
             "0.65490,0.66100,0.65760,0.66290,0.66260,0.66130,0.65760,0.65800",
             "0.65710,0.65620,0.65720,0.66490,0.66550,0.66500,0.66990,0.65560",
             "0.66110,0.65250,0.66520,0.65590,0.66650,0.66020,0.66290,0.66100",
             "0.66290,0.65950,0.65220,0.66670,0.65970,0.66130,0.66570,0.64960",
             "0.65840,0.65450,0.65840,0.66260,0.65870,0.66100,0.66230,0.65890",
             "0.66020,0.65930,0.66020,0.66080,0.66390,0.65540,0.66680,0.65510",
             "0.65790,0.66440,0.65330,0.66390,0.66780,0.65590,0.66180,0.65950",
             "0.65850,0.66260,0.65900,0.66190,0.66280,0.66370,0.66260,0.65540",
             "0.65710,0.65840,0.65890,0.65820,0.66680,0.66410,0.66470,0.65930",
             "0.65200,0.66420,0.66240,0.65690,0.66650,0.66000,0.66080,0.66440",
             "0.65360,0.65760,0.65970,0.65360,0.65790,0.66260,0.66160,0.65850",
             "0.65330,0.66050,0.65970,0.66210,0.66290,0.66290,0.66110,0.65150",
             "0.65660,0.66080,0.65840,0.66570,0.66630,0.65870,0.66210,0.65890",
             "0.65820,0.65710,0.65450,0.65760,0.65530,0.65980,0.66520,0.66060",
             "0.66130,0.65070,0.65660,0.65840,0.67150,0.66540,0.66390,0.65610",
             ]


def get_times_rel_ste():

  experiment = tb.data.experimental.ExperimentFromDev(sur_grads_tb['STE'])

  try:
    df = experiment.get_scalars()
  except grpc.RpcError as rpc_error:
    print('Couldn\'t fetch experiment: ' + experiment_id + ' got \
        error: ' + str(rpc_error))
    return None

  import pdb; pdb.set_trace()
  data = df[df['run'] == 'eval']
  return data[data['tag'] == 'accuracy']['value'].max()


def plot_surrogate():
  names = [x.split('_')[0] for x in sur_grads[0].split(',')]
  data = np.stack([x.split(',') for x in sur_grads[1:]])
  y = [float(x) * 100 if x != '' else np.nan for x in data.flatten(order='F')]
  x = np.repeat(np.arange(len(names)) + 1, 20)
  base_x = np.arange(len(names)) + 1

  np_data = np.array(
      [float(x) * 100 if x != '' else np.nan for x in data.flatten('F')
       ]).reshape((-1, 20))
  mu = np.nanmean(np_data, axis=1)
  sigma = np.nanstd(np_data, axis=1)

  font_size = 23

  fig, ax = plt.subplots(figsize=(15, 9))

  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  ax.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(5)

  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  ax.scatter(x / 2, y, marker='x', linewidths=5,
             s=180, color='blue', label='Observations')

  ax.scatter(base_x / 2, mu, marker='_', linewidths=5,
             s=840, color='red', label='Mean')

  ax.scatter(base_x / 2, mu + sigma, marker='_', linewidths=5,
             s=840, color='green', label='Std. Dev.')

  ax.scatter(base_x / 2, mu - sigma, marker='_',
             linewidths=5, s=840, color='green')

  plt.xticks(base_x / 2, names, rotation='horizontal')

  times = get_times_rel_ste()


  plt.legend(
      bbox_to_anchor=(0.5, 1.2),
      loc="upper center",
      ncol=3,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )
  ax.set_ylabel("Eval Accuracy (%)", fontsize=font_size, fontweight='bold')
  plt.tight_layout()
  plt.savefig('figures/surrogate_grads.png')
  plt.close()


def get_best_eval(experiment_id):
  experiment = tb.data.experimental.ExperimentFromDev(experiment_id)

  try:
    df = experiment.get_scalars()
  except grpc.RpcError as rpc_error:
    print('Couldn\'t fetch experiment: ' + experiment_id + ' got \
        error: ' + str(rpc_error))
    return None

  data = df[df['run'] == 'eval']
  return data[data['tag'] == 'accuracy']['value'].max()


def get_best_eval_and_size(experiment_id):
  experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
  try:
    df = experiment.get_scalars()
  except grpc.RpcError as rpc_error:
    print('Couldn\'t fetch experiment: ' + experiment_id + ' got \
        error: ' + str(rpc_error))
    return None, None

  data = df[df['run'] == 'eval']
  max_eval = data[data['tag'] == 'accuracy']['value'].max()
  if len(data[data['value'] == max_eval]) > 1:
    vals_at_step = data[data['step'] == int(
        data[data['value'] == max_eval]['step'].to_list()[0])]
  else:
    vals_at_step = data[data['step'] == int(
        data[data['value'] == max_eval]['step'])]
  size_mb = float(vals_at_step[vals_at_step['tag'] == 'weight_size']['value'])

  return max_eval, size_mb


def table_lr(res_dict):
  df = pd.DataFrame({'0.1': [None, None, None, None, None, None, None],
                     '0.01': [None, None, None, None, None, None, None],
                     '0.001': [None, None, None, None, None, None, None],
                     '0.0001': [None, None, None, None, None, None, None], })
  df.rename(index={0: '2', 1: '3', 2: '4', 3: '5',
                   4: '6', 5: '7', 6: '8'}, inplace=True)
  for k, v in res_dict.items():
    acc = get_best_eval(v)
    bit, lr = k.split('_')
    df[lr][bit] = acc

  return df


def table_init(res_dict):
  df = pd.DataFrame({'max': [None, None, None, None, None, None, None],
                     'doublemean': [None, None, None, None, None, None, None],
                     'gaussian': [None, None, None, None, None, None, None],
                     'p999': [None, None, None, None, None, None, None],
                     'p9999': [None, None, None, None, None, None, None],
                     'p99999': [None, None, None, None, None, None, None],
                     'p999999': [None, None, None, None, None, None, None],
                     })

  df.rename(index={0: 'max', 1: 'doublemean', 2: 'gaussian',
                   3: 'p999', 4: 'p9999', 5: 'p99999', 6: 'p999999'},
            inplace=True)
  for k, v in res_dict.items():
    acc = get_best_eval(v)
    w, a = k.split('_')
    df[w][a] = acc

  return df


def plot_line(ax, res_dict):
  x = []
  y = []
  for key, value in res_dict.items():
    if key == 'label':
      label = value
    elif key == 'params':
      pass
    else:
      acc_temp = get_best_eval(value)
      if acc_temp is not None and acc_temp > .15:
        y.append(1 - acc_temp)
        x.append(key * res_dict['params'] / 8_000_000)

  ax.plot(x, y, marker='x', label=label, ms=20, markeredgewidth=5, linewidth=5)


def plot_mixed(ax, res_dict):
  x = []
  y = []
  for key, value in res_dict.items():
    if key == 'label':
      label = value
    else:
      acc_t, size_t = get_best_eval_and_size(value)
      if acc_t is None:
        continue
      y.append(1 - acc_t)
      x.append(size_t)

  ax.plot(x, y, marker='x', label=label, ms=20, markeredgewidth=5, linewidth=5)


def plot_comparison(name):
  font_size = 26

  fig, ax = plt.subplots(figsize=(32, 13.8))

  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  ax.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(5)

  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  # Competitors.
  # add PROFIT numbers (https://arxiv.org/pdf/2008.04693.pdf)
  for competitor_name, competitor_data in competitors.items():
    ax.plot(competitor_data['size_mb'], competitor_data['eval_err'],
            label=competitor_data['name'],
            marker='.', ms=20, markeredgewidth=5, linewidth=5,
            alpha=competitor_data['alpha'])

  # Our own.
  # plot_line(ax, enets_fp32_t2)
  plot_line(ax, enets_int8)

  # plot_line(ax, enet0_dynamic_init_double_mean_t2)
  # plot_line(ax, enet0_static_init_double_mean_t2)
  # plot_line(ax, enet0_dynamic_init_gaussian_t2)
  # plot_line(ax, enet0_static_init_gaussian_t2)

  plot_line(ax, enet0_lr_best)

  xv = [1.1886465, 1.337227313, 1.485808125, 1.634388938, 1.78296975,
        2.080131375, 2.228712188, 2.377293, 2.525873813, 2.674454625,
        2.823035438, 2.97161625]
  yv = [0.6473, 0.4253, 0.3882, 0.3739, 0.3638, 0.3314,
        0.3241, 0.3184, 0.3068, 0.286, 0.2812, 0.2795]
  ax.plot(xv, yv, marker='x', label="Mixed EfficientNet0",
          ms=20, markeredgewidth=5, linewidth=5, color='red')
  # plot_line(ax, enet0_dynamic_lsq)

  # plot_mixed(ax, enet0_mixed)
  # plot_mixed(ax, enet0_mixed_8)
  # plot_mixed(ax, enet0_mixed_4)
  ax.set_xscale('log')
  ax.set_xlabel("Network Size (MB)", fontsize=font_size, fontweight='bold')
  ax.set_ylabel("Eval Error (%)", fontsize=font_size, fontweight='bold')
  plt.legend(
      bbox_to_anchor=(1., 1.),
      loc="upper left",
      ncol=1,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )
  plt.tight_layout()
  plt.savefig(name)
  plt.close()


def plot_bits_vs_acc(list_of_dicts, name):
  font_size = 26

  fig, ax = plt.subplots(figsize=(13, 9.8))
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  ax.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(5)

  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  for v in list_of_dicts:
    plot_line(ax, v)

  ax.set_xlabel("Bits", fontsize=font_size, fontweight='bold')
  ax.set_ylabel("Eval Accuracy (%)", fontsize=font_size, fontweight='bold')
  plt.legend(
      bbox_to_anchor=(0.5, 1.2),
      loc="upper center",
      ncol=2,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )
  plt.tight_layout()
  plt.savefig(name)
  plt.close()


if __name__ == '__main__':
  major_ver, minor_ver, _ = version.parse(tb.__version__).release
  assert major_ver >= 2 and minor_ver >= 3, \
      "This notebook requires TensorBoard 2.3 or later."
  print("TensorBoard version: ", tb.__version__)

  plot_surrogate()
  plot_comparison('figures/overview.png')

  # # df_dynamic_lr = table(enet0_dynamic_lr)
  # df_static_lr = table_lr(enet0_static_lr)
  # df_static_init_4b = table_init(enet0_static_init_4b)
  # df_static_init_3b = table_init(enet0_static_init_3b)

  # # print("Dynamic Quant")
  # # print(df_dynamic_lr.to_csv())

  # print("LR Static Quant")
  # print(df_static_lr.to_csv())

  # print("Init Static Quant 4b")
  # print(df_static_init_4b.to_csv())

  # print("Init Static Quant 3b")
  # print(df_static_init_3b.to_csv())

  # plot_bits_vs_acc([enet0_dynamic_init_max, enet0_dynamic_init_double_mean,
  #                   enet0_dynamic_init_gaussian], 'figures/dynamic_init.png')

  # plot_bits_vs_acc([enet0_static_init_max, enet0_static_init_double_mean,
  #                   enet0_static_init_gaussian], 'figures/static_init.png')

  # plot_bits_vs_acc([enet0_dynamic_init_max_surrogate,
  # enet0_static_init_max_surrogate, enet0_dynamic_init_max,
  # enet0_static_init_max], 'figures/surrogate.png')

  # plot_bits_vs_acc([enet0_dynamic_init_double_cos,
  #                   #enet0_dynamic_init_double_cos_sur,
  #                   #enet0_dynamic_init_double_cos_psg,
  #                   enet0_dynamic_init_double_cos_ewgs],
  #                  'figures/psg_ewgs.png')
