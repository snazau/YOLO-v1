maxpool_description = {
    "type": "maxpool",
    "params": {
        "kernel_size": 2,
        "stride": 2,
    }
}

backbone_config = [
    {
        "type": "conv",
        "params": {
            "kernel_size": 7,
            "output_filter_amount": 64,
            "stride": 2,
            "padding": 3,
        }
    },
    maxpool_description,
    {
        "type": "conv",
        "params": {
            "kernel_size": 3,
            "output_filter_amount": 192,
            "stride": 1,
            "padding": 1,
        }
    },
    maxpool_description,
    {
        "type": "conv",
        "params": {
            "kernel_size": 1,
            "output_filter_amount": 128,
            "stride": 1,
            "padding": 0,
        }
    },
    {
        "type": "conv",
        "params": {
            "kernel_size": 3,
            "output_filter_amount": 256,
            "stride": 1,
            "padding": 1,
        }
    },
    {
        "type": "conv",
        "params": {
            "kernel_size": 1,
            "output_filter_amount": 256,
            "stride": 1,
            "padding": 0,
        }
    },
    {
        "type": "conv",
        "params": {
            "kernel_size": 3,
            "output_filter_amount": 512,
            "stride": 1,
            "padding": 1,
        }
    },
    maxpool_description,
    {
        "type": "repeated_block",
        "repeats_amount": 4,
        "layers_descriptions": [
            {
                "type": "conv",
                "params": {
                    "kernel_size": 1,
                    "output_filter_amount": 256,
                    "stride": 1,
                    "padding": 0,
                }
            },
            {
                "type": "conv",
                "params": {
                    "kernel_size": 3,
                    "output_filter_amount": 512,
                    "stride": 1,
                    "padding": 1,
                }
            }
        ]
    },
    {
        "type": "conv",
        "params": {
            "kernel_size": 1,
            "output_filter_amount": 512,
            "stride": 1,
            "padding": 0,
        }
    },
    {
        "type": "conv",
        "params": {
            "kernel_size": 3,
            "output_filter_amount": 1024,
            "stride": 1,
            "padding": 1,
        }
    },
    maxpool_description,
    {
        "type": "repeated_block",
        "repeats_amount": 2,
        "layers_descriptions": [
            {
                "type": "conv",
                "params": {
                    "kernel_size": 1,
                    "output_filter_amount": 512,
                    "stride": 1,
                    "padding": 0,
                }
            },
            {
                "type": "conv",
                "params": {
                    "kernel_size": 3,
                    "output_filter_amount": 1024,
                    "stride": 1,
                    "padding": 1,
                }
            }
        ]
    },
    {
        "type": "conv",
        "params": {
            "kernel_size": 3,
            "output_filter_amount": 1024,
            "stride": 1,
            "padding": 1,
        }
    },
    {
        "type": "conv",
        "params": {
            "kernel_size": 3,
            "output_filter_amount": 1024,
            "stride": 2,
            "padding": 1,
        }
    },
    {
        "type": "conv",
        "params": {
            "kernel_size": 3,
            "output_filter_amount": 1024,
            "stride": 1,
            "padding": 1,
        }
    },
    {
        "type": "conv",
        "params": {
            "kernel_size": 3,
            "output_filter_amount": 1024,
            "stride": 1,
            "padding": 1,
        }
    },
]
