class HParamsView(object):
    def __init__(self, d):
        self.__dict__ = d


def create_hparams(**kwargs):
    """Create spk_embedder hyperparameters. Parse nondefault from given string."""

    hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "iters_per_checkpoint": 10000,
        "seed": 16807,
        "fp16_run": False,
        "cudnn_enabled": True,
        "cudnn_benchmark": True,
        "output_directory": "dyadic_bcn3_restart",  # Directory to save checkpoints.
        "log_directory": 'log',
        "checkpoint_path": 'mnt/techfak_compute/genea/genea23/genea_challenge_2023-main/baselines/2023_ivi_baseline-main/Tacotron2/dyadic_bcn3/ckpt/checkpoint_600000.pt',
        "warm_start": False,
        "device": 0,
        "n_acoustic_feat_dims": 168,
        "dyadic": True,
        "warmup_period": 100,
        "sequence_length": 200,
        ################################
        # Model Parameters             #
        ################################

        ################################
        # Audio Encoder                #
        ################################
        "model_monadic": HParamsView({
            "warmup_period": 100,
            "text_attention": HParamsView({
                "conv_in": 638,
                "kernel_size": 1,
                "embed_dim": 512,
                "num_heads": 8,
                "dropout": 0.5,
                "bias": False,
                "feedforward": HParamsView({
                    "in_dim": 512,
                    "lin_dim": 256,
                    "out_dim": 512,
                })
            }),
            "audio_attention": HParamsView({
                "conv_in": 216,
                "kernel_size": 1,
                "embed_dim": 512,
                "num_heads": 8,
                "dropout": 0.5,
                "bias": False,
                "feedforward": HParamsView({
                    "in_dim": 512,
                    "lin_dim": 256,
                    "out_dim": 512,
                })
            }),
            "combined_attention": HParamsView({
                "conv_in": 216,
                "kernel_size": 1,
                "embed_dim": 512,
                "num_heads": 8,
                "dropout": 0.5,
                "bias": False,
                "feedforward": HParamsView({
                    "in_dim": 512,
                    "lin_dim": 256,
                    "out_dim": 512,
                })
            }),
            "combined_attention_lin_dim": 1024,
            "combined_attention_embed_dim": 512,
            "combined_attention_num_heads": 8,
            "combined_attention_dropout": 0.5,

            "combined_feedforward": HParamsView({
                "in_dim": 512,
                "attention_heads": 8,
                "lin_dim": 256,
                "out_dim": 512
            }),
            "fein": HParamsView({
                "conv_in": 854,
                "gesture_in": 336,
                "conv_dim": 512,
                "conv_out": 256,
                "kernel_size": 1,
                "lin_dim": 256,
                "gesture_attention": HParamsView({
                    "conv_in": 512,
                    "kernel_size": 1,
                    "embed_dim": 512,
                    "num_heads": 8,
                    "dropout": 0.5,
                    "bias": False,
                    "feedforward": HParamsView({
                        "in_dim": 512,
                        "lin_dim": 256,
                        "out_dim": 512,
                    })
                })
            }),
            "control_network": HParamsView({
                "control_in": 512,
                "control_dim": 256,
                "control_out": 128,
                "controllable_joints": [3, 21, 18, 18, 9, 9]
            })
        }),

        "model_dyadic": HParamsView({
            "warmup_period": 100,
            "text_attention": HParamsView({
                "conv_in": 638,
                "kernel_size": 1,
                "embed_dim": 512,
                "num_heads": 8,
                "dropout": 0.5,
                "bias": False,
                "feedforward": HParamsView({
                    "in_dim": 512,
                    "lin_dim": 256,
                    "out_dim": 512,
                })
            }),
            "audio_attention": HParamsView({
                "conv_in": 216,
                "kernel_size": 1,
                "embed_dim": 512,
                "num_heads": 8,
                "dropout": 0.5,
                "bias": False,
                "feedforward": HParamsView({
                    "in_dim": 512,
                    "lin_dim": 256,
                    "out_dim": 512,
                })
            }),
            "combined_attention": HParamsView({
                "conv_in": 512,
                "kernel_size": 1,
                "embed_dim": 512,
                "num_heads": 8,
                "dropout": 0.5,
                "bias": False,
                "feedforward": HParamsView({
                    "in_dim": 512,
                    "lin_dim": 256,
                    "out_dim": 512,
                })
            }),
            "combined_attention_lin_dim": 1024,
            "combined_attention_embed_dim": 512,
            "combined_attention_num_heads": 8,
            "combined_attention_dropout": 0.5,

            "combined_feedforward": HParamsView({
                "in_dim": 512,
                "attention_heads": 8,
                "lin_dim": 256,
                "out_dim": 512
            }),
            "fein": HParamsView({
                "conv_in": 854,
                "gesture_in": 336,
                "conv_dim_ta":1024,
                "conv_dim": 512,
                "conv_out": 512,
                "kernel_size": 1,
                "lin_dim": 256,
                "gesture_attention": HParamsView({
                    "conv_in": 512,
                    "kernel_size": 1,
                    "embed_dim": 512,
                    "num_heads": 8,
                    "dropout": 0.5,
                    "bias": False,
                    "feedforward": HParamsView({
                        "in_dim": 512,
                        "lin_dim": 256,
                        "out_dim": 512,
                    })
                })
            }),
            "control_network": HParamsView({
                "control_in": 512,
                "control_dim": 256,
                "control_out": 128,
                "controllable_joints": [3, 21, 6, 6, 18, 48, 18, 48]
            })
        }),
        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate": False,
        "learning_rate": 5e-5,
        "leaning_rate_disc": 1e-4,
        "weight_decay": 0.01,
        "grad_clip_thresh": 1.0,
        "batch_size": 128,
        "mask_padding": True,  # set spk_embedder's padded outputs to padded values
        "mel_weight": 1,
        "gate_weight": 0,
        "vel_weight": 1,
        "pos_weight": 0.01,
        "add_l1_losss": False
    }

    for key, val in kwargs.items():
        if key in hparams:
            hparams[key] = val
        else:
            raise ValueError('The hyper-parameter %s is not supported.' % key)

    hparams_view = HParamsView(hparams)

    return hparams_view
