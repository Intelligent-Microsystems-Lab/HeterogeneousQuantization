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
    3: '9nwVcMdPRRGJtjDjrZpCeg',
    4: 'kX80UjhnQxWdwEdzQDTNzQ',
    5: 'KMwHB3liQM2WR7jSzE7nNQ',
    6: 'mx3Gs5p4T0S9KSBXeFqAvg',
    7: 'yZQjrnLBS9iG3DFeUzoaPg',
    8: 'vq1yDQlLSgSbNH0he3QE5A',
    'params': 4652008,
    'label': 'EfficientNet-Lite0 (3-8 Bits)',
}


enet0_static_lr = {
    '2_0.0001': '',
    '2_0.001': '',
    '2_0.01': '',
    '2_0.1': '',
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

enet0_static_init = {
    'max_max': '7I2DKoqsQrqAH2O63D4eig',
    'max_doublemean': '4aRGN3vNSA6o0qA4TaRcPw',
    'max_gaussian': 'U1fqXpfwROG6hpWWiwqQPQ',
    'max_p999': 'NsYjCPURSMiMujFrOC0Vzg',
    'max_p9999': '2Zemsxh4Tzu1VikEx4djKg',
    'max_p99999': 'eznienrKQrmQg1dnnJpblQ',
    'max_p999999': 'DpDDJ9BxSXmQDPOioaMpQA',

    'doublemean_max': 'ulnMJxo3TveGAtD3ADrpwg',
    'doublemean_doublemean': '15vphyrbSyuLlgL83m9KKg',
    'doublemean_gaussian': 'U6R18149SyagJ54Mo9tEiQ',
    'doublemean_p999': 'BYwFYZOKREWLZg6w0UDVpA',
    'doublemean_p9999': '1ca0JiOESaG0jRbExPiDqQ',
    'doublemean_p99999': 'dNVOpjxJTd6XNGOERfalrw',
    'doublemean_p999999': '9x4exLtVSlOyn8Uj4HxXBg',

    'gaussian_max': 'hn06OO2YRtGyQcxtYYjOLw',
    'gaussian_doublemean': 'jDPh2vqwRmGa0Baio2AalA',
    'gaussian_gaussian': 'iLAKzaklSQGurdFMxDtJOg',
    'gaussian_p999': 'x0CEYBxTQp6ZeMe8ZDH84Q',
    'gaussian_p9999': 'v7Ggrm3ERf6iWz8Dh0NZMw',
    'gaussian_p99999': '21av70YJTYqErM24ZwO3zg',
    'gaussian_p999999': 'hEQrEEZ1QOeEQ32Hjsm5zg',

    'p999_max': 'ymAa2fj5RsmZTMZfveQGkw',
    'p999_doublemean': 'g10BBnYfQQeavWO10V5qWA',
    'p999_gaussian': 'dPvPT6s2TDiVOk9RyBIxqw',
    'p999_p999': 'gspOQVWzQ32zOubnpQEl0g',
    'p999_p9999': 'zC4T3s6AT12ELjo087ymnA',
    'p999_p99999': 'qjnwaYsQRlq1bVY8rlVQWA',
    'p999_p999999': 'ZeP2eOK6TdiBG48sOal4hA',

    'p9999_max': 'pofbEKP0QGaM6Fq5FqNfPA',
    'p9999_doublemean': 'vxcXeVQ4QoKyMeN0XwGYCw',
    'p9999_gaussian': 'gEPlRpt3QdGmYGKpy7udeQ',
    'p9999_p999': 'sZc04MPcRWGqH0rEFDKGQg',
    'p9999_p9999': 'j5cpofVtRXGPtkObZYvO9w',
    'p9999_p99999': 'Fymq6Rb2TDGsQHokjFEE3g',
    'p9999_p999999': 'wgFAziqNTEePmv38bAuvcg',

    'p99999_max': '0Eji5GCKS0W0KMaqzxCAhg',
    'p99999_doublemean': 'xHNPvrj7SEyRW7ZRbB0JsA',
    'p99999_gaussian': '5gHRwwTyRIqTJCywO9D54w',
    'p99999_p999': 'YYEm3vSXT4SehZOsGQQIJg',
    'p99999_p9999': 'nelFQpgbTla0ZZxqmBt9NQ',
    'p99999_p99999': 'xzrG4uwRTXKqe059leSRpA',
    'p99999_p999999': 'HxgPEoyvRJ2I6iGxCKVbAA',

    'p999999_max': 'QxxRtDxmTJidP0ObX3S05w',
    'p999999_doublemean': '5u5agSr6TKuAd5yQrXXV7w',
    'p999999_gaussian': '9mRYmMFLT4i0KVJaU5Rz2w',
    'p999999_p999': 'oUQVLuqoREG3WnnkLWfJfQ',
    'p999999_p9999': '7k5gVecvQKqLBZYOH79Ymw',
    'p999999_p99999': 'lsDDCzMXTaOG7OQ0pWWzBw',
    'p999999_p999999': '5vwtp6xYTrarCnesX7Z2mA',
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

    'hawqv2_squeeze': {
        # https://arxiv.org/abs/1911.03852
        'eval_err': [1 - 0.6838],
        'size_mb': np.array([1.07]),
        'name': 'HAWQ-V2 SqueezeNext',
        'alpha': 1.,
    },

    'hawqv2_inceptionv3': {
        # https://arxiv.org/abs/1911.03852
        'eval_err': [1 - 0.7568],
        'size_mb': np.array([7.57]),
        'name': 'HAWQ-V2 Inception-V3',
        'alpha': 1.,
    },

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

    'haq_mobilev2': {
        # https://arxiv.org/pdf/1811.08886.pdf
        'eval_err': [1 - 0.6675, 1 - 0.7090, 1 - 0.7147],
        'size_mb': np.array([.95, 1.38, 1.79]),
        'name': 'HAQ MobileNetV2',
        'alpha': 1.,
    },

    'haq_resnet50': {
        # https://arxiv.org/pdf/1811.08886.pdf
        'eval_err': [1 - 0.7063, 1 - 0.7530, 1 - 0.7614],
        'size_mb': np.array([6.30, 9.22, 12.14]),
        'name': 'HAQ ResNet50',
        'alpha': 1.,
    },
}


def get_best_eval(experiment_id):
  experiment = tb.data.experimental.ExperimentFromDev(experiment_id)

  try:
    df = experiment.get_scalars()
  except grpc.RpcError as rpc_error:
    print('Couldn\'t fetch experiment: ' + experiment_id + ' got error: '
          + str(rpc_error))
    return None

  data = df[df['run'] == 'eval']
  return data[data['tag'] == 'accuracy']['value'].max()


def get_best_eval_and_size(experiment_id):
  experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
  df = experiment.get_scalars()

  data = df[df['run'] == 'eval']
  max_eval = data[data['tag'] == 'accuracy']['value'].max()
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
            3: 'p999', 4: 'p9999', 5: 'p99999', 6: 'p999999'}, inplace=True)
  for k, v in res_dict.items():
    acc = get_best_eval(v)
    bit, lr = k.split('_')
    df[lr][bit] = acc

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
      y.append(1 - get_best_eval(value))
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

  plot_mixed(ax, enet0_mixed)

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

  plot_comparison('figures/overview.png')

  # df_dynamic_lr = table(enet0_dynamic_lr)
  df_static_lr = table_lr(enet0_static_lr)
  df_static_init = table_init(enet0_static_init)
  # print("Dynamic Quant")
  # print(df_dynamic_lr.to_csv())
  print("LR Static Quant")
  print(df_static_lr.to_csv())

  print("Init Static Quant")
  print(df_static_init.to_csv())

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
